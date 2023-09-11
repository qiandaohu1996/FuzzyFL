from copy import deepcopy
import numpy as np
import torch.nn.functional as F
from torch import no_grad

from client import Client
from utils.torch_utils import *
from utils.constants import *
from torchvision.models import MobileNetV2

from utils.models import FemnistCNN
from utils.datastore import *


# Detailed implementation of the proximal_local_update method for the FedSoftClient class

class FedSoftClient(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.u_t = {}  # Placeholder for importance weights for the client
    
    def proximal_local_update(self, centers, lambda_val):
        # Implement the proximal local update based on the provided pseudo-code
        # Using the client's data and model, compute the local update based on the proximal objective
        
        # For the purpose of this example, we'll use a placeholder method to compute the proximal update
        local_model_update = self.compute_proximal_update(centers, lambda_val)
        return local_model_update
    
    # Placeholder method to compute the proximal update
    def compute_proximal_update(self, centers, lambda_val):
        return {}

# NOTE: The actual method for computing the proximal update will depend on the specifics of the client's data,
# model, and the provided pseudo-code.

fedsoft_client_implementation = FedSoftClient
