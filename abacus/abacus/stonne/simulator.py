"""Congigure stonne"""
import os

class Simulator(object):
    path:str = ""
    _ms_size:int
    reduce_network_type:str
    dn_bw:int
    rn_bw:int
    controller_type:str
    sparsity_ratio:float 
    tune:bool = False # A variable which to set if you want to use config
    
    @property
    def ms_size(self):
        return self._ms_size
    
    @ms_size.setter
    def ms_size(self, size: int):
        self._ms_size = size
        self.create_config_file(self.path)

    def edit_config(
        self,
        ms_size:int,
        reduce_network_type:str,
        dn_bw:int,
        rn_bw:int,
        controller_type:str,
        sparsity_ratio:int = 0, 
        tune:bool = False
    ):
        self._ms_size = ms_size
        print("set", reduce_network_type)
        self.reduce_network_type = reduce_network_type
        self.dn_bw = dn_bw
        self.rn_bw = rn_bw
        self.controller_type = controller_type 
        self.sparsity_ratio =sparsity_ratio
        self.tune = tune

    def create_config_file(self, path:str):
        """
        This will create a config file at a desired location
        """
        if path == "" and self.path == "":
            path = os.getcwd()
            self.path = path + "/stonne_config.cfg"

        with open(self.path, "w") as f:
            f.write("[MSNetwork]\n")
            f.write(f"ms_size={self.ms_size}\n")
            f.write("[ReduceNetwork]\n")
            f.write(f'type="{self.reduce_network_type}"\n')
            f.write("[SDMemory]\n")
            f.write(f"dn_bw={self.dn_bw}\n")
            f.write(f"rn_bw={self.rn_bw}\n")
            f.write(f'controller_type="{self.controller_type}"\n')

def config_simulator(
    ms_size:int,
    reduce_network_type:str,
    dn_bw:int,
    rn_bw:int,
    controller_type:str,
    sparsity_ratio:int = 0, 
    path:str = "",
    tune:bool = False
    ):

    """
    This function builds a new STONNE config
    """

    architecture.edit_config(
        ms_size,
        reduce_network_type,
        dn_bw,
        rn_bw,
        controller_type ,
        sparsity_ratio,
        tune
    )
    architecture.create_config_file(path = path)

architecture = Simulator()