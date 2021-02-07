import os

# T_M=[x]: Number of output neurons
# T_N=[x]: Batch size
# T_K=[x]: Number of input neurons

class FCTileConfig(object):
    def __init__(self):
        self.path: str
        self.T_N : int 
        self.T_S : int 
        self.T_K : int 

    def generate_basic_tile_config(self):
        self.T_N = 1 
        self.T_S = 1 
        self.T_K = 1 
        self.path = (
            os.getcwd() 
            + "/fc_tile_config_"
            + str(self.T_N) 
            + str(self.T_S) 
            + str(self.T_K) 
            + ".txt" 
        )


    def create_tile_file(self):

        with open(self.path, "w") as f:
            f.write(f'tile_type="FC"\n')
            f.write(f"T_N={self.T_N}\n")
            f.write(f"T_S={self.T_S}\n")
            f.write(f"T_K={self.T_K}\n")

fc_tiles = FCTileConfig()
