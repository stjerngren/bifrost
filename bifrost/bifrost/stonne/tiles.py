import os

class TileConfig(object):

    def __init__(self):
        self.path :str 
        self.tile_type :str
        self.T_R : int 
        self.T_S : int 
        self.T_C : int 
        self.T_G : int 
        self.T_K : int 
        self.T_N : int 
        self.T_X : int 
        self.T_Y : int 

    def edit_tile_config(
        self,
        path : str,
        tile_type : str,
        T_R : int,
        T_S : int,
        T_C : int,
        T_G : int,
        T_K : int,
        T_N : int,
        T_X : int,
        T_Y : int,
    ):
        self.path = (
            path 
            + "/tile_config_"
            + str(T_R)
            + str(T_S)
            + str(T_C)
            + str(T_G)
            + str(T_K)
            + str(T_N)
            + str(T_X)
            + str(T_Y)   
            +".txt"
        )
        self.tile_type = tile_type
        self.T_R = T_R
        self.T_S = T_S
        self.T_C = T_C
        self.T_G = T_G
        self.T_K = T_K
        self.T_N = T_N
        self.T_X = T_X
        self.T_Y = T_Y   

    def create_tile_file(self):

        with open(self.path, "w") as f:
            f.write(f'tile_type="{self.tile_type}"\n')
            f.write(f"T_R={self.T_R}\n")
            f.write(f"T_S={self.T_S}\n")
            f.write(f"T_C={self.T_C}\n")
            f.write(f"T_G={self.T_G}\n")
            f.write(f"T_K={self.T_K}\n")
            f.write(f"T_N={self.T_N}\n")
            f.write(f"T_X'={self.T_X}\n")
            f.write(f"T_Y'={self.T_Y}\n")   


tiles = TileConfig()

def config_tile_file(
    tile_type : str,
    T_R : int,
    T_S : int,
    T_C : int,
    T_G : int,
    T_K : int,
    T_N : int,
    T_X : int,
    T_Y : int,
    path : str = "",
):
    if path == "":
        path = os.getcwd()

    tiles.edit_tile_config(
        path,
        tile_type,
        T_R,
        T_S,
        T_C,
        T_G,
        T_K,
        T_N,
        T_X,
        T_Y,
    )

def generate_basic_tile_config(
    tile_type :str,
    R: int,
    S: int,
    C:int,
    K:int,
    G:int,
    X:int,
    Y:int,
    strides:int,
    ):

    config_tile_file(
        tile_type,
        R, 
        S, 
        C, 
        G, 
        K, 
        1, 
        (X - R + strides) // strides,
        (Y - S + strides) // strides,
    )

    return R * S *C *G *K *1 * ((X - R + strides) // strides) *((Y - S + strides) // strides)
