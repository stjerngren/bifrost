from typing import List
from .tile_tuner import create_conv_tile_tuning_space

class TuningParameters(object):
    def __init__(
        self
    ) -> None:
        self.tune_convolutions_tile:bool = False
        self.tune_dense:bool = False
        self.dense_num: int = 1
        self.conv_num: int = 1
        self.conv_tile_knobs:List = []
        self.dense_tile_knobs:List = []
    
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

    def create_knobs(self)->List:
        """
        Based on set tuning parameters, create all the knobs

        Returns
        -------
        all_knobs: List[Tuple(str,List[Object])]
            A list of the tuning knobs

        """
        all_knobs = []
        all_knobs.extend(self.conv_tile_knobs)
        all_knobs.extend(self.dense_tile_knobs)
        return all_knobs

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
        self.conv_tile_knobs = create_conv_tile_tuning_space(R,S,C,K,G,X,Y,strides,self.conv_num)


tune_parameters = TuningParameters()
