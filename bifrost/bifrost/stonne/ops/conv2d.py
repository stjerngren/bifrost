""" 
Register everything to do with conv2d
"""
import tvm
from tvm import te, relay, autotvm
from tvm.topi import generic
import tvm.relay.op as _op
from tvm.relay.op.strategy.generic import *
import os
from ..simulator import architecture
from ..tiles import tiles
#from tvm.topi.nn.utils import traverse_inline

# Register the compute schedule for stonne conv2d
@autotvm.register_topi_compute("conv2d_stonne_nchw.x86")
def conv2d_stonne_nchw(
    cfg,
    data, 
    kernel, 
    strides, 
    padding, 
    dilation, 
    groups=1, 
    layout="NCHW", 
    out_dtype="float32"
    
    ):    
    """
    Compute conv2d using STONNE
    """


    # Define tuning space
    cfg.define_knob("ms_size", [8,16,64])
    if False:
        knobs = [
            ("a",[12,3,45,6,7]),
            ("b",[6,7]),
            ("c",[12,6,7])
            ]
        for knob in knobs:
            cfg.define_knob(*knob)

    # If the architecture is being tuned, write to the file with the
    # following name
    tuning_name = "ms_size_" + str(cfg['ms_size'])

    if architecture.tune:
        # Change the architecture to the new settings
        architecture.ms_size = cfg['ms_size']
        # Create a temporary file for tuning config
        architecture.create_config_file(name_config="ms_size_" + str(cfg['ms_size']))

        #TODO Add tuning file to architecture
        # tuning_file = archtecture
    
    path = architecture.path
    sparsity_ratio = architecture.sparsity_ratio
    dirname = os.path.dirname(__file__)
    costs_path = os.path.join(dirname, "../data/costs.json")
    # Extract data from 
    N, C, H, W = get_const_tuple(data.shape)
    output_channels, _, kernel_height, kernel_width = get_const_tuple(kernel.shape)
    
    if N>1:
        raise ValueError("STONNE does not support a batch size greater than one")
    """
    Translate variables names to STONNE taxonomy
    -R: Number of flter rows
    -S: Number of filter columns
    -C: Number of filter and input channels
    -K: Number of filters and output channels
    -G: Number of groups
    -N: Number of inputs (Only 1 is supported so far)
    -X: Number of input rows
    -Y: Number of input columns
    -X_: Number of output columns
    -Y_: Number of output columns
    """
    X = H  
    Y = W 
    R = kernel_height
    S = kernel_width
    K = output_channels
    G = 1
    N = 1 
    pad_x = padding[0] 
    pad_y = padding[1]
    
    # Calculate the output shape
    X_:int = ((X + 2 * pad_x - dilation[0] * (R - 1) - 1) // strides[0]) + 1
    Y_:int = ((Y + 2 * pad_y - dilation[1] * (S - 1) - 1) // strides[0]) + 1
    
    return te.extern(
            (N,K,X_, Y_),
            [data,kernel],
            lambda ins, outs: tvm.tir.call_packed(
                "tvm.contrib.stonne.conv2d.forward",  
                path,              # [0]
                R,                 # [1]
                S,                 # [2]
                C,                 # [3]
                K,                 # [4]
                G,                 # [5]
                N,                 # [6]
                X,                 # [7]
                Y,                 # [8]
                X_,                # [9]    
                Y_,                # [10]    
                strides[0],        # [11]      
                pad_x,             # [12]    
                pad_y,             # [13]    
                tiles.path,        # [14]     
                sparsity_ratio,    # [15]    
                architecture.tune, # [16]
                tuning_name,       # [17]
                costs_path,        # [18]
                ins[0],            # [19]
                ins[1],            # [20]
                outs[0],           # [21]


            ),
            name = "s",
            dtype = out_dtype
    )

# Use the genric schedule
@autotvm.register_topi_schedule("conv2d_stonne_nchw.x86")
def schedule_conv2d_stonne(cfg, outs):
    """Create schedule for conv2d_nhwc"""
    cfg.add_flop(2)

    return te.create_schedule([x.op for x in outs])


# Override the conv2d x86 strategy to add STONNE support in the libs
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


    """
    This is the only part which changes
    """
    if "stonne" in target.libs:
        if layout == "NCHW":
            strategy.add_implementation(
                    wrap_compute_conv2d(conv2d_stonne_nchw),
                    wrap_topi_schedule(schedule_conv2d_stonne),
                    name="conv2d_stonne.x86",
            )
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

