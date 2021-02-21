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
from bifrost.stonne.simulator import config_simulator, architecture

import os

architecture.ms_size = 128
architecture.rn_bw = 8
architecture.dn_bw = 128
architecture.load_tile_config(conv_cfg_paths = ["/Users/axelstjerngren/uni/Year4/ProjectLevel4/level-4-project/bifrost/bifrost_temp/test.txt"])
architecture.create_config_file()

# Let’s create a very simple network for demonstration.
# It consists of convolution, batch normalization, and ReLU activation.
logging.basicConfig(level=logging.DEBUG)  # to dump TVM IR after fusion
out_channels = 2
batch_size = 1

# Let's create a very simple network for demonstration.
# It consists of convolution, batch normalization, and ReLU activation.

data = relay.var("data", relay.TensorType((batch_size, 2, 50, 50), "float32"))
weight = relay.var("weight")
bn_gamma = relay.var("bn_gamma")
bn_beta = relay.var("bn_beta")
bn_mmean = relay.var("bn_mean")
bn_mvar = relay.var("bn_var")

simple_net = relay.nn.conv2d(
    data=data, weight=weight, kernel_size=(5, 5), channels=out_channels, padding=(1, 1)
)

simple_net = relay.nn.relu(simple_net)
simple_net = relay.nn.conv2d(simple_net, weight=weight, kernel_size=(5, 5), channels=out_channels, padding=(1, 1)
)
simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)



data_shape = (batch_size, 2, 50, 50)
net, params = testing.create_workload(simple_net)

# Generate the data to resuse with both llvm and llvm stonne
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

# Build and run with llvm backend

target = "llvm"
lib = relay.build_module.build(net, target, params=params)

ctx = tvm.context(target, 0)
module = runtime.GraphModule(lib["default"](ctx))
module.set_input("data", data)
module.run()
out_shape = (batch_size, out_channels, 46, 46)
out = module.get_output(0, tvm.nd.empty(out_shape))
out_llvm = out.asnumpy()

print(out_llvm)
# Build and run with llvm backend, but this time use the
# stonne conv2d ops

target = "llvm -libs=stonne"
lib = relay.build_module.build(net, target, params=params)

ctx = tvm.context(target, 0)
module = runtime.GraphModule(lib["default"](ctx))
module.set_input("data", data)
module.run()
out_shape = (batch_size, out_channels, 46, 46)
out = module.get_output(0, tvm.nd.empty(out_shape))
out_stonne = out.asnumpy()

print(np.all(np.round(out_stonne, 4) == np.round(out_llvm, 4)))

print(np.round(out_stonne,4) == np.round(out_llvm, 4))

print(out_llvm.shape)
print(out_stonne.shape)
print(data.shape)

print(out_llvm)
print(out_stonne)

