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
import abacus.abacus

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
#simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
#simple_net = relay.nn.relu(simple_net)
#simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)

data_shape = (batch_size, 2, 10, 10)
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
out_shape = (batch_size, out_channels, 10, 10)
out = module.get_output(0, tvm.nd.empty(out_shape))
out_llvm = out.asnumpy()
# Build and run with llvm backend, but this time use the
# stonne conv2d ops
 
target = "llvm -libs=stonne"
lib = relay.build_module.build(net, target, params=params)

ctx = tvm.context(target, 0)
module = runtime.GraphModule(lib["default"](ctx))
module.set_input("data", data)
module.run()
out_shape = (batch_size, out_channels, 10, 10)
out = module.get_output(0, tvm.nd.empty(out_shape))
out_stonne = out.asnumpy()

print(np.all(np.round(out_stonne, 4) == np.round(out_llvm, 4)))

print(np.round(out_stonne,4) == np.round(out_llvm, 4))

print(out_llvm.shape)
print(out_stonne.shape)
print(data.shape)

print(out_llvm)
print(out_stonne)
