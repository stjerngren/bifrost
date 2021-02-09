from .tile_tuner import create_conv_tile_tuning_space

class TuningParameters(object):
    def __init__(
        self
    ) -> None:
        self.tune_convolutions_tile:bool = False
        self.tune_dense:bool = False
        self.dense_num: int = 1
        self.conv_num: int = 1
    
    def parameters(self,
        tune_convolutions_tile:bool = False,
        tune_dense_tile:bool = False,
        tune_architecture: bool = False,
        dense_num: int = 1,
        conv_num: int = 1,
        tune_accumulation_buffer:bool = False,
        tune_sparsity_ratio:bool = False,
        sparsity_ratio_num: int = 1,
        tune_ms_size:bool = True,
        tune_rn_bw:bool = True,
        tune_dn_bw:bool = True,
        tune_reduce_network_type:bool = True,
        tune_ms_network_type:bool = True,
    ):
        pass

    def create_knobs(self):
        return 

    def tune_everything():
        pass

    def conv_tile(self,
        R: int,
        S: int,
        C:int,
        K:int,
        G:int,
        X:int,
        Y:int,
        strides:int,
    ):
        create_conv_tile_tuning_space(R,S,C,K,G,X,Y,strides,conv_num)

tune_parameters = TuningParameters()
