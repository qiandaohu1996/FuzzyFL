from copy import deepcopy

import numpy as np

from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from aggregator import Aggregator
from utils.my_profiler import calc_exec_time, segment_timing
from utils.torch_utils import *
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
# from utils.fuzzy_cluster import *
# from finch import FINCH

from learners.learner import *
from learners.learners_ensemble import *
from utils.fuzzy_utils import *


calc_time = True
SAMPLING_FLAG = False

# Detailed implementation of estimate_importance_weights and compute_aggregation_weights methods

class FedSoftAggregator(Aggregator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.importance_weights = {}  # Placeholder for importance weights u_{ks}^t
    
    def estimate_importance_weights(self):
        # Send centers to all clients
        # NOTE: The method for sending centers to clients depends on the actual implementation 
        # and communication setup between the server and clients.
        # For the purpose of this example, we'll use a placeholder method.
        centers = self.send_centers_to_clients()
        
        # Clients compute their importance weights
        # NOTE: The method for clients to compute importance weights depends on the client's data 
        # and the provided pseudo-code.
        # For the purpose of this example, we'll use a placeholder method.
        self.importance_weights = self.get_importance_weights_from_clients()
        
    def compute_aggregation_weights(self):
        # Compute aggregation weights v_{sk}^t as described in the pseudo-code
        v_sk_t = {}
        for s, weights in self.importance_weights.items():
            v_sk_t[s] = {}  # Placeholder for aggregation weights for cluster s
            total_weight = sum(weights.values())
            for k, u_ks in weights.items():
                v_sk_t[s][k] = u_ks * len(weights) / total_weight
        
        return v_sk_t
    
    # Placeholder methods for sending centers to clients and getting importance weights from clients
    def send_centers_to_clients(self):
        return {}
    
    def get_importance_weights_from_clients(self):
        return {}
    
    # Using the existing methods as mentioned
    def aggregate_client_models(self):
        return self.average_learners()
    
    def select_clients(self):
        return self.update_client()

# NOTE: The actual methods for sending centers to clients and getting importance weights from clients
# will depend on the specifics of the communication setup between the server and clients.

fedsoft_aggregator_implementation = FedSoftAggregator
