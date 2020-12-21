
import tvm
from tvm import te, autotvm ,relay, rpc
import numpy as np
from tvm.contrib import graph_runtime as runtime
from tvm.relay import testing
import logging
import random


# Import this add stonne as an x86 co-processor
import abacus.abacus
from abacus.abacus.stonne.simulator import config_simulator
from abacus.abacus.tuner.stone_builder import StonneLocalBuilder, StonneLocalRunner

from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner


config_simulator(
    ms_size=16,
    reduce_network_type="ASNETWORK",
    dn_bw=8,
    rn_bw=8,
    controller_type="MAERI_DENSE_WORKLOAD",
    tune = True
)


# TODO: Add a way to configure STONNE
out_channels = 2
batch_size = 1

# Let’s create a very simple network for demonstration.
# It consists of convolution, batch normalization, and ReLU activation.

data = relay.var("data", relay.TensorType((batch_size, 2, 10, 10), "float32"))
weight = relay.var("weight")
bn_gamma = relay.var("bn_gamma")
bn_beta = relay.var("bn_beta")
bn_mmean = relay.var("bn_mean")
bn_mvar = relay.var("bn_var")

simple_net = relay.nn.conv2d(
    data=data, weight=weight, kernel_size=(3, 3), channels=out_channels, padding=(1, 1)
)

#simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)

data_shape = (batch_size, 2, 10, 10)
net, params = testing.create_workload(simple_net)

# Generate the data to resuse with both llvm and llvm stonne
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

# Build and run with llvm backend
logging.basicConfig(level=logging.DEBUG)  # to dump TVM IR after fusion


# Build and run with llvm backend, and use the
# stonne conv2d ops
target = "llvm --libs=stonne"

mod, params = testing.create_workload(simple_net)
log_file = "test.log"

tuning_options = {
    "log_filename": log_file,
    "tuner": "random",
    "early_stopping": None,
    "measure_option": autotvm.measure_option(
        builder=StonneLocalBuilder(),
        runner=StonneLocalRunner(
            number=1,
            repeat=10,
            min_repeat_ms=0,
            enable_cpu_cache_flush=True
        ),
    ),
}
batch_size = 1

# You can skip the implementation of this function for this tutorial.
def tune_kernels(
    tasks, measure_option, tuner="gridsearch", early_stopping=None, log_filename="tuning.log"
):

    for i, task in enumerate(tasks):
        print(task)
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(task, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(task)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial = len(task.config_space)
        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename),
            ],
        )

# Use graph tuner to achieve graph level optimal schedules
# Set use_DP=False if it takes too long to finish.
def tune_graph(graph, dshape, records, opt_sch_file, use_DP=True):
    target_op = [
        relay.op.get("nn.conv2d"),
    ]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)

if __name__ == "__main__":
    remote = rpc.LocalSession()

    tasks = autotvm.task.extract_from_program(
            mod, 
            target=target, 
            params=params, 
            ops=(relay.op.get("nn.conv2d"),)
    )

    tune_kernels(tasks, tuning_options["measure_option"])

    #tune_graph(mod["main"], data_shape, log_file, graph_opt_sch_file)
#
    ## compile kernels with graph-level best records
    #with autotvm.apply_graph_best(graph_opt_sch_file):
    #    print("Compile...")
    #    with tvm.transform.PassContext(opt_level=3):
    #        lib = relay.build_module.build(mod, target=target, params=params)
#
    #    # upload parameters to device
    #    ctx = tvm.cpu()
    #    data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
    #    module = runtime.GraphModule(lib["default"](ctx))
    #    module.set_input(input_name, data_tvm)
#
    #    # evaluate
    #    print("Evaluate inference time cost...")
    #    ftimer = module.module.time_evaluator("run", ctx, number=100, repeat=3)
    #    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    #    print(
    #        "Mean inference time (std dev): %.2f ms (%.2f ms)"
    #        % (np.mean(prof_res), np.std(prof_res))
    #    )