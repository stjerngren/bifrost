""" 
Register everything to do with dense
"""
import tvm
from tvm import te, relay, autotvm
from tvm.topi import generic
import tvm.relay.op as _op
from tvm.relay.op.strategy.generic import *
import os
from ..simulator import architecture
from tvm.auto_scheduler import is_auto_scheduler_enabled
from tvm.te import SpecializedCondition
from .. import fc_tiles

@autotvm.register_topi_compute("dense_stonne.x86")
def dense_stonne(cfg,data, weight, units=None, out_dtype=""):
    """Dense operator
    Applies a linear transformation

    .. math::

    `Y = X * W^T`

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator,
        of shape `(d_1, d_2, ..., d_n, units_in)`.

    weight : tvm.relay.Expr
        The weight expressions, 2-D matrix,
        of shape `(units, units_in)`.

    units : int, optional
        Number of hidden units of the dense transformation.

    out_dtype : str, optional
        Specifies the output data type for mixed precision dense,
        of shape `(d_1, d_2, ..., d_n, units)`.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    # Get the name to store the costs
    dirname = os.path.dirname(__file__)
    costs_path = os.path.join(dirname, "../data/costs.json")

    M, K = get_const_tuple(data.shape)
    N, _ = get_const_tuple(weight.shape)

    # Define tuning space
    if architecture.tune:
        
        # Generate the different fc tile options
        if architecture.tuner.tune_fc_tile:
            architecture.tuner.fc_tile()

        # Get and register the tuning knobs
        knobs = architecture.tuner.create_knobs()
        for knob in knobs:
            cfg.define_knob(*knob)
        
        # Create the architecture files
        architecture.config(cfg)

    # Choose tiles
    elif architecture.manual_tile_paths:
        architecture.set_manual_tile_config("FC")

    if architecture.tune and not architecture.manual_tile_paths and not architecture.tuner.tune_fc_tile:    
        architecture.fc_tiles_path = architecture.fc_tiles.generate_basic_tile_config()

    return te.extern(
            (M,N),
            [data,weight],
            lambda ins, outs: tvm.tir.call_packed(
                "tvm.contrib.stonne.dense",  
                architecture.path, # [0] Architecture file
                M,                 # [1] Batch size
                K,                 # [2] Number of input neurons
                N,                 # [3] Number of output neurons
                architecture.fc_tiles_path,   # [4] Tiles path
                architecture.sparsity_ratio,  # [5]
                architecture.print_stats,     # [6] Create stats output files
                architecture.tune, # [7] Enable if tuning
                "tuning_name",     # [8]
                costs_path,        # [9]
                ins[0],            # [10] Data
                ins[1],            # [11] Weight
                outs[0],           # [12] Output
 
            ),
            name = "d",
            dtype = out_dtype
    )
    

@autotvm.register_topi_schedule("dense_stonne.x86")
def schedule_dense_stonne(cfg, outs):
    return te.create_schedule([x.op for x in outs])


@dense_strategy.register("cpu")
def dense_strategy_cpu(attrs, inputs, out_type, target):
    """dense x86 strategy"""
    strategy = _op.OpStrategy()
    m, _ = inputs[0].shape
    same_type = inputs[0].dtype == inputs[1].dtype == out_type.dtype
    dtype = inputs[0].dtype
    u8s8s32 = dtype == "uint8" and inputs[1].dtype == "int8" and out_type.dtype == "int32"
    strategy.add_implementation(
        wrap_compute_dense(topi.x86.dense_nopack),
        wrap_topi_schedule(topi.x86.schedule_dense_nopack),
        name="dense_nopack.x86",
        plevel=10,
    )


    if "stonne" in target.libs:
        strategy.add_implementation(
            wrap_compute_dense(dense_stonne),
            wrap_topi_schedule(schedule_dense_stonne),
            name="dense_stonne.x86",
            plevel=12,
        )

    if is_auto_scheduler_enabled():
        strategy.add_implementation(
            wrap_compute_dense(topi.nn.dense, need_auto_scheduler_layout=True),
            naive_schedule,
            name="dense.generic",
            plevel=11,
        )

    if "cblas" in target.libs:
        with SpecializedCondition(same_type and dtype in ["float32", "float64"]):
            strategy.add_implementation(
                wrap_compute_dense(topi.x86.dense_cblas),
                wrap_topi_schedule(topi.x86.schedule_dense_cblas),
                name="dense_cblas.x86",
                plevel=13,
            )
    if "mkl" in target.libs:
        with SpecializedCondition(same_type and dtype in ["float32", "float64"] or u8s8s32):
            strategy.add_implementation(
                wrap_compute_dense(topi.x86.dense_mkl),
                wrap_topi_schedule(topi.x86.schedule_dense_mkl),
                name="dense_mkl.x86",
                plevel=14,
            )
    if "mkldnn" in target.libs:
        with SpecializedCondition(same_type and dtype == "float32"):
            strategy.add_implementation(
                wrap_compute_dense(topi.x86.dense_mkldnn),
                wrap_topi_schedule(topi.x86.schedule_dense_mkldnn),
                name="dense_mkldnn.x86",
                plevel=15,
            )
    with SpecializedCondition(m >= 16):
        # this implementation may not be well-optimized, so use plevel=5 for now.
        strategy.add_implementation(
            wrap_compute_dense(topi.x86.dense_pack),
            wrap_topi_schedule(topi.x86.schedule_dense_pack),
            name="dense_pack.x86",
            plevel=5,
        )
    return strategy