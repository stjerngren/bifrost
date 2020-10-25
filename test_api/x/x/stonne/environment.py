import tvm
from typing import Dict


def get_env():
    pass



class Simulator(object):
    config : Dict[str,int]

    def __init__(self, config: Dict[str,int]):
        self.config = config

    #TODO add target
    @property
    def target(self):
        return None

    @property
    def target_host(self):
        return None

def config_simulator(config: Dict[str,int]) -> Simulator:
    """
    This function builds a STONNE config and returns a simulator object
    """
    return Simulator(config)
