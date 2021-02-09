from typing import List
from .tile_tuner import create_conv_tile_tuning_space

class TuningParameters(object):
    def __init__(
        self
    ) -> None:
        self.tune_convolutions_tile:bool = False
        self.tune_fc_tile:bool = False
        self.fc_num: int = 5
        self.conv_num: int = 3
        self.conv_tile_knobs:List = []
        self.fc_tile_knobs:List = []
        self.tune_accumulation_buffer: bool = False
        self.tune_sparsity_ratio: bool = False
        self.sparsity_ratio_num:int = 1
        self.tune_reduce_network_type:bool = False
        self.tune_ms_network_type:bool = False   
        self.tune_ms_size:bool = False
    
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
        all_knobs.extend(self.fc_tile_knobs)
        if self.tune_accumulation_buffer:
            all_knobs.append(("accumulation_buffer",[True,False]))
        if self.tune_reduce_network_type:
            all_knobs.append(("reduce_network_type",["ASNETWORK","FENETWORK"]))
        if self.tune_ms_size:
            all_knobs.append(("ms_size",[32,64,128]))
        self.conv_tile_knobs = []
        self.fc_tile_knobs = []
        return all_knobs

    def tune_maeri_all(self):
        self.tune_accumulation_buffer = False
        self.tune_reduce_network_type = False
        self.tune_ms_size = True
        self.tune_convolutions_tile = True
        self.tune_fc_tile = True

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
