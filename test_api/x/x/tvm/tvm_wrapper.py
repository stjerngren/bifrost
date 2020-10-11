import tvm.relay as relay

def load_module(model, shape):
    """Thin wrapper of tvm.relay.frontend.from_pytorch

    Supports only pytorch intially, look into adding more type later

    See Also
    --------
    tvm.relay.frontend.from_pytorch : The original TVM's pytorch loading module
    """

    mod, params = relay.frontend.from_pytorch(model, shape)


    return mod, params 