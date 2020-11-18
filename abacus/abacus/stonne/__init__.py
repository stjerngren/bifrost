import tvm

from .connect_stonne import load_lib

# Register STONNE library
_LIB = load_lib()

print("yup")
# Register the ops
from . import ops

