"""Namespace for supporting Relay operators on STONNE."""

import tvm
from tvm import te
from tvm import topi

from tvm.relay.op import op as reg
from tvm.relay.op import strategy as _strategy
from tvm.relay.op.op import OpPattern, OpStrategy


@_strategy.conv2d_transpose_strategy.register("stonne")
def conv2d_transpose_strategy_stonne(attrs, inputs, out_type, target):
    return None

