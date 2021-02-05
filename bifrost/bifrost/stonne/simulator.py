"""Congigure stonne"""
import os

# Define a new error for improper configs
class ConfigError(Exception):
    pass

class Simulator(object):
    def __init__(self):
        self._path: str = "" # Internal path without filename
        self.path:str = "" # Path with filename
        self._ms_size:int = 16
        self._ms_rows:int = 16
        self._ms_cols:int = 16
        self._reduce_network_type:str = "ASNETWORK"
        self._ms_network_type:str = "Linear"
        self._dn_bw:int = 8
        self._rn_bw:int = 8
        self._controller_type:str = "MAERI_DENSE_WORKLOAD"
        self.sparsity_ratio:float = 0
        self.tune:bool = False # A variable which to set if you want to use config
        self.print_stats:bool = False # Create output stats for stonne
        self._knobs:tuple = () # An empty tuple for now
        self._accumulation_buffer_enabled = 1

    @property
    def ms_size(self):
        return self._ms_size
    @ms_size.setter
    def ms_size(self, size: int):
        # Use bit manipulation magic to check if power of two
        if (size & (size-1) == 0) and size != 0:
            self._ms_size = size
        else:
            raise ConfigError("ms_size has to be a power of two!")
    
    @property
    def reduce_network_type(self):
        return self._reduce_network_type
    @reduce_network_type.setter
    def reduce_network_type(self, type: str):
        if type in ("ASNETWORK","FENETWORK","TEMPORALRN"):
            self._reduce_network_type = type
        else:
            raise ConfigError("Reduce network has to be ASNETWORK, FENETWORK, or TEMPORALRN") 
   
    @property
    def dn_bw(self):
        return self._dn_bw
    @dn_bw.setter
    def dn_bw(self, size: int):
        # Use bit manipulation magic to check if power of two
        if (size & (size-1) == 0) and size != 0:
            self._dn_bw= size
        else:
            raise ConfigError("dn_bw has to be a power of two!")
    
    @property
    def rn_bw(self):
        return self._dn_bw
    @rn_bw.setter
    def rn_bw(self, size: int):
        # Use bit manipulation magic to check if power of two
        if (size & (size-1) == 0) and size != 0:
            self._rn_bw= size

        else:
            raise ConfigError("dn_bw has to be a power of two!")
    
    @property
    def rn_bw(self):
        return self._dn_bw
    @rn_bw.setter
    def rn_bw(self, size: int):
        # Use bit manipulation magic to check if power of two
        if (size & (size-1) == 0) and size != 0:
            self._rn_bw= size
        else:
            raise ConfigError("dn_bw has to be a power of two!")

    @property
    def controller_type(self):
        return self._controller_type
    @controller_type.setter
    def controller_type(self, type: str):
        if type in ("MAERI_DENSE_WORKLOAD","SIGMA_SPARSE_GEMM","MAGMA_SPARSE_DENSE","TPU_OS_DENSE"):
            self._controller_type = type
            if type == "TPU_OS_DENSE":
                self.reduce_network_type = "TEMPORALRN"
                self.ms_network_type = "OS_MESH"
                self.dn_bw = self.ms_rows + self.ms_cols
                self.rn_bw = self.ms_rows * self.ms_cols
                print("TPU selected, automatically set:")
                print("Reduce network to TEMPORALRN")
                print("MS network to OS_MESH")
                print(f"dn_bw to {self.dn_bw}")
                print(f"dn_bw to {self.rn_bw}")
        else:
            raise ConfigError("SDMemory controller type has to be MAERI_DENSE_WORKLOAD,SIGMA_SPARSE_GEMM,MAGMA_SPARSE_DENSE or TPU_OS_DENSE") 

    @property
    def ms_network_type(self):
        return self._ms_network_type
    @ms_network_type.setter
    def ms_network_type(self, type: str):
        if type in ("LINEAR","OS_MESH"):
            self._ms_network_type= type
        else:
            raise ConfigError("LINEAR","OS_MESH") 

    @property
    def accumulation_buffer_enabled(self):
        return self._accumulation_buffer_enabled
    @accumulation_buffer_enabled.setter
    def accumulation_buffer_enabled(self, enabled: bool):
        if type(enabled) ==bool:
            if enabled:
                self._accumulation_buffer_enabled = 1
            else:
                self._accumulation_buffer_enabled = 0
        else:
            raise ConfigError("Accumulation buffer should be a boolean value")



    def edit_config(
        self,
        ms_size:int,
        reduce_network_type:str,
        ms_network_type:str,
        dn_bw:int,
        rn_bw:int,
        controller_type:str,
        sparsity_ratio:int = 0,
        ms_rows = 16,
        ms_cols = 16, 
        tune:bool = False,
        print_stats:bool = False,
        accumulation_buffer_enabled:bool = True,

    ):
        """
        Set a whole new architectuer config
        """
        self.ms_size = ms_size
        self.ms_cols = ms_cols
        self.ms_rows = ms_rows
        self.reduce_network_type = reduce_network_type
        self.dn_bw = dn_bw
        self.rn_bw = rn_bw
        self.controller_type = controller_type 
        self.sparsity_ratio =sparsity_ratio
        self.tune = tune
        self.print_stats = print_stats
        self.ms_network_type = ms_network_type
        self.accumulation_buffer_enabled = accumulation_buffer_enabled

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
            f.write(f'type="{self.ms_network_type}"\n')
            if self.ms_network_type == "LINEAR":
                f.write(f"ms_size={self.ms_size}\n")
            else:
                f.write(f"ms_rows={self.ms_rows}\n")
                f.write(f"ms_cols={self.ms_cols}\n")
            f.write("[ReduceNetwork]\n")
            f.write(f'type="{self.reduce_network_type}"\n')
            f.write(f'accumulation_buffer_enabled="{self.accumulation_buffer_enabled}"\n')
            f.write("[SDMemory]\n")
            f.write(f"dn_bw={self.dn_bw}\n")
            f.write(f"rn_bw={self.rn_bw}\n")
            f.write(f'controller_type="{self.controller_type}"\n')
        print("New config created at ", self.path, self.ms_size)

def config_simulator(
    ms_size:int,
    reduce_network_type:str,
    ms_network_type:str,
    accumulation_buffer_enabled:bool,
    dn_bw:int,
    rn_bw:int,
    controller_type:str,
    sparsity_ratio:int = 0, 
    path:str = "",
    tune:bool = False,
    ms_rows = 16,
    ms_cols = 16,
    print_stats = False,
    ):

    """
    This function builds a new STONNE config
    """

    architecture.edit_config(
        ms_size,
        reduce_network_type,
        ms_network_type,
        dn_bw,
        rn_bw,
        controller_type,
        sparsity_ratio,
        ms_rows,
        ms_cols,
        tune,
        print_stats,
        accumulation_buffer_enabled,

    )
    architecture.create_config_file(path = path)

architecture = Simulator()