REGISTRY = {}

from .basic_controller import BasicMAC
from .central_basic_controller import CentralBasicMAC
from .opt_controller import OptMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["opt_mac"] = OptMAC
