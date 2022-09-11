from .basic_controller import BasicMAC
from .vffac_controller import VffacMAC
from .immac_controller import ImmacMAC
from .marnd_controller import MarndMAC


REGISTRY = {
    "basic_mac": BasicMAC,
    'vffac_mac': VffacMAC,
    'immac_mac': ImmacMAC,
    'marnd_mac': MarndMAC
}


