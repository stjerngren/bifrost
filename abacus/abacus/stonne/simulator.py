"""Congigure stonne"""
import os

class Simulator(object):
    path:str = ""
    ms_size:int
    reduce_network_type:str
    dn_bw:int
    rn_bw:int
    controller_type:str
    sparsity_ratio:float 

    def edit_config(
        self,
        ms_size:int,
        reduce_network_type:str,
        dn_bw:int,
        rn_bw:int,
        controller_type:str,
        sparsity_ratio:int = 0, 
    ):
        self.ms_size = ms_size
        self.reduce_network_type = reduce_network_type
        self.dn_bw = dn_bw
        self.rn_bw = rn_bw
        self.controller_type = controller_type 
        self.sparsity_ratio =sparsity_ratio

    def create_config_file(self, path:str):
        """
        This will create a config file at a desired location
        """
        if path == "":
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
    )
    architecture.create_config_file(path = path)

architecture = Simulator()