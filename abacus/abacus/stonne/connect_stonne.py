
import ctypes
import os



def load_lib():
    """
    Register the stonne connection library into TVM
    """

    # Find library based on relative paths
    dirname = os.path.dirname(__file__)
    stonne_conv2d = os.path.join(dirname, "../../lib/conv_forward.so")

    print(dirname)
    print(stonne_conv2d)
    # load in as global so the global extern symbol is visible to other dll.
    lib = ctypes.CDLL(stonne_conv2d, ctypes.RTLD_GLOBAL)
    return lib


