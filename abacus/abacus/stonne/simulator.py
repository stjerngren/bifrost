"""Congigure stonne"""


class Simulator(object):

    # TODO: Create objects for these...
    simulation_file:str  
    tiles_path:str       
    sparsity_ratio:float  

    def __init__(
        self,
        simulation_file:str   = "/Users/axelstjerngren/uni/Year4/ProjectLevel4/level-4-project/simulator_default/test.cfg",
        tiles_path:str        = "/Users/axelstjerngren/uni/Year4/ProjectLevel4/level-4-project/simulator_default/tile_configuration_conv1.txt",
        sparsity_ratio:float  =  0.0 
    ):
        self.simulation_file = simulation_file
        self.tiles_path = tiles_path
        self.sparsity_ratio = sparsity_ratio

def config_simulator():
    """
    This function builds a new STONNE config
    """
    