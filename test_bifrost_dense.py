"""
Try to load STONNE as an external lib for conv2d

The smae way cuDNN or cuBLAS would be used with tvm.relay

"""

import tvm
from tvm import te, autotvm ,relay
import numpy as np
from tvm.contrib import graph_runtime as runtime
from tvm.relay import testing
import logging
import random

# Import this add stonne as an x86 co-processor
import bifrost
from bifrost.stonne.simulator import config_simulator

config_simulator(
    ms_size=16,
    reduce_network_type="ASNETWORK",
    ms_network_type= "LINEAR",
    accumulation_buffer_enabled = True,
    dn_bw=8,
    rn_bw=8,
    controller_type="MAERI_DENSE_WORKLOAD",
)


out_channels = 2
batch_size = 1

# Let’s create a very simple network for demonstration.

data = relay.var("data", shape=(1, 2))
weight = relay.var("weight", shape=(6, 2))

simple_net = relay.nn.dense(
    data=data, weight=weight
)

data_shape = (1,2)
net, params = testing.create_workload(simple_net)

# Generate the data to resuse with both llvm and llvm stonne
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")


# Build and run with llvm backend
logging.basicConfig(level=logging.DEBUG)  # to dump TVM IR after fusion

target = "llvm"
lib = relay.build_module.build(net, target, params=params)

ctx = tvm.context(target, 0)
module = runtime.GraphModule(lib["default"](ctx))
module.set_input("data", data)
module.run()

out_shape = (1,6)
out = module.get_output(0, tvm.nd.empty(out_shape))
out_llvm = out.asnumpy()
print(out_llvm)
# Build and run with llvm backend, but this time use the
 #stonne conv2d ops
 
target = "llvm -libs=stonne"
lib = relay.build_module.build(net, target, params=params)

ctx = tvm.context(target, 0)
module = runtime.GraphModule(lib["default"](ctx))
module.set_input("data", data)
module.run()
out_shape = (1,6)
out = module.get_output(0, tvm.nd.empty(out_shape))
out_stonne = out.asnumpy()
print(out_stonne)

print(np.all(np.round(out_stonne, 4) == np.round(out_llvm, 4)))

print(np.round(out_stonne,4) == np.round(out_llvm, 4))

print(out_llvm.shape)
print(out_stonne.shape)
print(data.shape)

print(out_llvm)
print(out_stonne)