"""Congigure stonne"""
import os

class Simulator(object):
    def __init__(self):
        self._path: str = "" # Internal path without filename
        self.path:str = "" # Path with filename
        self._ms_size:int = 16
        self.reduce_network_type:str = "ASNETWORK"
        self.dn_bw:int = 8
        self.rn_bw:int = 8
        self.controller_type:str = "MAERI_DENSE_WORKLOAD"
        self.sparsity_ratio:float = 0
        self.tune:bool = False # A variable which to set if you want to use config
        self.knobs:tuple = () # An empty tuple for now
    @property
    def ms_size(self):
        return self._ms_size
    
    @ms_size.setter
    def ms_size(self, size: int):
        self._ms_size = size

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
        """
        Set a whole new architectuer config
        """
        self._ms_size = ms_size
        self.reduce_network_type = reduce_network_type
        self.dn_bw = dn_bw
        self.rn_bw = rn_bw
        self.controller_type = controller_type 
        self.sparsity_ratio =sparsity_ratio
        self.tune = tune

    def create_config_file(self, path:str = "", name_config:str = None):
        """
        This will create a config file at a desired location

        Parameters
        ----------
        
        path : str (optional)
            The path for the config to be saved at. If none provided, the new 
            config will be created in the new working directory
        
        name_config : str (optional)
            An optional name for the specific cofig
        """
        if path == "": 
            path = os.getcwd()
            self._path = path
            if name_config:
                name = "/stonne_config_" + name_config + ".cfg"
            else:
                name = "/stonne_config.cfg"
            self.path = path + name

        # write arcitecture to file
        with open(self.path, "w") as f:
            f.write("[MSNetwork]\n")
            f.write(f"ms_size={self.ms_size}\n")
            f.write("[ReduceNetwork]\n")
            f.write(f'type="{self.reduce_network_type}"\n')
            f.write("[SDMemory]\n")
            f.write(f"dn_bw={self.dn_bw}\n")
            f.write(f"rn_bw={self.rn_bw}\n")
            f.write(f'controller_type="{self.controller_type}"\n')
        print("New config created at ", self.path, self.ms_size)

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