"""
Try to load STONNE as an external lib for conv2d

The smae way cuDNN or cuBLAS would be used with tvm.relay

"""
from StonneUtils import getTileFileFromConvDimensions

import tvm
from tvm import te, autotvm ,relay
import numpy as np
from tvm.contrib import graph_runtime as runtime
from tvm.relay import testing
import logging
import torch.nn as nn
from tvm.topi.util import get_const_tuple

from tvm.relay.op.strategy.generic import *
import tvm.relay.op as _op

# Start by registering stonne conv2d 


simulation_file='test.cfg'
tiles_path='tiles/accumulation_buffer/128_mses/'
sparsity_ratio=0.0

@autotvm.register_topi_compute("conv2d_stonne.x86")
def conv2d_stonne(
    cfg, data, kernel, strides, padding, dilation, groups=1, layout="NCHW", out_dtype="float32"
):
    """Compute conv2d using stonne library"""
    N, C, H, W = get_const_tuple(data.shape)
    tile_file_1 = getTileFileFromConvDimensions(tiles_path, 3, 3, 1, 1)


    print("This is the kernel", kernel)
    return te.extern(
            data.shape,
            [data,kernel],
            lambda ins, outs: tvm.tir.call_packed(
                nn.SimulatedConv2d( 3,  3, 3,  simulation_file, tile_file_1, sparsity_ratio, stride=strides, padding=padding),
            ),
            out_dtype,
    )






@conv2d_strategy.register("cpu")
def conv2d_strategy_cpu(attrs, inputs, out_type, target):
    """conv2d x86 strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")
    if "stonne" in target.libs:
        strategy.add_implementation(
                wrap_compute_conv2d(conv2d_stonne),
                wrap_topi_schedule(topi.generic.schedule_conv2d_nchw),

                name="conv2d_stonne.x86",
        )
        print("Hi")
    else:
        if groups == 1:
            if layout == "NCHW":
                assert kernel_layout == "OIHW"
                if topi.x86.is_int8_hw_support(data.dtype, kernel.dtype):
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.x86.conv2d_nchw_int8),
                        wrap_topi_schedule(topi.x86.schedule_conv2d_nchw_int8),
                        name="conv2d_nchw_int8.x86",
                    )
                else:
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.x86.conv2d_nchw),
                        wrap_topi_schedule(topi.x86.schedule_conv2d_nchw),
                        name="conv2d_nchw.x86",
                    )
            elif _NCHWc_matcher.match(layout):  # check if layout is NCHWxc
                assert _OIHWio_matcher.match(kernel_layout)  # check if kernel is OIHWio
                return conv2d_NCHWc_strategy_cpu(attrs, inputs, out_type, target)
            elif layout == "NHWC":
                assert kernel_layout == "HWIO"
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.conv2d_nhwc),
                    wrap_topi_schedule(topi.x86.schedule_conv2d_nhwc),
                    name="conv2d_nhwc.x86",
                )
            elif layout == "HWCN":
                assert kernel_layout == "HWIO"
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.conv2d_hwcn),
                    wrap_topi_schedule(topi.generic.schedule_conv2d_hwcn),
                    name="conv2d_hwcn.generic",
                )
            else:
                raise RuntimeError("Unsupported conv2d layout {} for x86".format(layout))
        elif is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups):
            if layout == "NCHW":
                assert kernel_layout == "OIHW"
                channel_multiplier = get_const_tuple(inputs[1].shape)[1]
                if channel_multiplier == 1 and dilation_h == 1 and dilation_w == 1:
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.x86.depthwise_conv2d_nchw),
                        wrap_topi_schedule(topi.x86.schedule_depthwise_conv2d_nchw),
                        name="depthwise_conv2d_nchw.x86",
                    )
                else:
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.nn.depthwise_conv2d_nchw),
                        wrap_topi_schedule(topi.generic.schedule_depthwise_conv2d_nchw),
                        name="depthwise_conv2d_nchw.generic",
                    )
            elif _NCHWc_matcher.match(layout):  # check if layout is NCHWxc
                assert _OIHWio_matcher.match(kernel_layout)  # check if kernel is OIHWio
                return depthwise_conv2d_NCHWc_strategy_cpu(attrs, inputs, out_type, target)
            elif layout == "NHWC":
                assert kernel_layout == "HWOI"
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.depthwise_conv2d_nhwc),
                    wrap_topi_schedule(topi.generic.schedule_depthwise_conv2d_nhwc),
                    name="depthwise_conv2d_nhwc.generic",
                )
            else:
                raise RuntimeError("Unsupported depthwise_conv2d layout {}".format(layout))
        else:  # group_conv2d
            if layout == "NCHW":
                assert kernel_layout == "OIHW"
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.group_conv2d_nchw, has_groups=True),
                    wrap_topi_schedule(topi.generic.schedule_group_conv2d_nchw),
                    name="group_conv2d_nchw.generic",
                )
            elif layout == "NHWC":
                assert kernel_layout == "HWIO"
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.group_conv2d_nhwc, has_groups=True),
                    wrap_topi_schedule(topi.generic.schedule_group_conv2d_nhwc),
                    name="group_conv2d_nhwc.generic",
                )
            else:
                raise RuntimeError("Unsupported group_conv2d layout {}".format(layout))
    return strategy




out_channels = 16
batch_size = 1

# Let’s create a very simple network for demonstration. It consists of convolution, batch normalization, and ReLU activation.

data = relay.var("data", relay.TensorType((batch_size, 3, 224, 224), "float32"))
weight = relay.var("weight")
bn_gamma = relay.var("bn_gamma")
bn_beta = relay.var("bn_beta")
bn_mmean = relay.var("bn_mean")
bn_mvar = relay.var("bn_var")

simple_net = relay.nn.conv2d(
    data=data, weight=weight, kernel_size=(3, 3), channels=out_channels, padding=(1, 1)
)
simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
simple_net = relay.nn.relu(simple_net)
simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)

data_shape = (batch_size, 3, 224, 224)
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
out_shape = (batch_size, out_channels, 224, 224)
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
out_shape = (batch_size, out_channels, 224, 224)
out = module.get_output(0, tvm.nd.empty(out_shape))
out_stonne = out.asnumpy()

print(np.all(out_stonne == out_llvm))