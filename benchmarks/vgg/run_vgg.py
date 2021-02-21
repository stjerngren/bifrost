
import torch
from bifrost.stonne.simulator import config_simulator, architecture
from vgg import vgg11_torch as torch_model
from vgg import input_batch as data
import numpy as np

import tvm
from tvm import te, autotvm ,relay, rpc
from tvm.contrib import graph_runtime as runtime
import bifrost

architecture.ms_size = 128
torch_model.eval()
trace = torch.jit.trace(torch_model, data).eval()


with autotvm.apply_history_best("bifrost_temp/vgg.log"):
    

    mod, params = relay.frontend.from_pytorch(
        trace, [("trace", data.shape)])
    target = "llvm -libs=stonne"
    
    lib = relay.build(mod, target=target, params=params)

    ctx = tvm.context(target, 0)
    module = runtime.GraphModule(lib["default"](ctx))
    module.set_input("trace", data)
    module.run()

    test = module.get_output(0)
print(test)