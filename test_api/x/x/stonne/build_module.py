from typing import Dict

class Simulator(object):
    config : Dict[str,int]

    def __init__(self, config: Dict[str,int]):
        self.config = config

def config_simulator(config: Dict[str,int]) -> Simulator:
    """
    This function builds a STONNE config and returns a simulator object
    """
    return Simulator(config)



print(config_simulator({"r":1}))