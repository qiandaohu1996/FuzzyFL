from copy import deepcopy
import numpy as np
import torch.nn.functional as F
from torch import no_grad

from client.client import Client
from utils.torch_utils import *
from utils.constants import *
from torchvision.models import MobileNetV2

from utils.models import FemnistCNN
# from utils.datastore import *

class FedSoftClient(Client):
    def __init__(
        self, 
        learners_ensemble, 
        idx, 
        train_iterator, 
        val_iterator, 
        test_iterator, 
        logger, 
        local_steps=2, 
        tune_locally=False,
    ):
        super(FedSoftClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            idx=idx,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
        )
        
        self.sigma = 1e-2
        self.cluster_weights=None
        self.momentum=0.8
        self.pre_cluster_weights=None
        
    def init_cluster_weights(self,n_clusters):
        self.cluster_weights = torch.ones(n_clusters, dtype=torch.float32,device=self.learners_ensemble.device)

    def update_cluster_weights(self,cluster_learners):
        """
        Update the cluster weights based on the distance from cluster models (learners)
        """
        
        n_clusters=len(cluster_learners)
        self.init_cluster_weights(n_clusters)
        data_point_counts = torch.zeros(n_clusters, dtype=torch.float32,device=self.learners_ensemble.device)
        
        for batch in self.train_iterator:
            x, y, indices = batch
            batch_losses = cluster_learners.compute_gradients_and_loss(batch)
            for j in range(len(indices)):
                min_loss_idx = torch.argmin(torch.tensor([loss[j] for loss in batch_losses]))
                data_point_counts[min_loss_idx] += 1
                
 
        # 归一化
        data_point_counts /= data_point_counts.sum()
        # 使用 torch.clamp 进行剪切，确保每个权重不小于 self.sigma
        self.cluster_weights = torch.clamp(data_point_counts, min=self.sigma)
        # 再次归一化以确保剪切后的权重之和为1
        self.cluster_weights /= self.cluster_weights.sum()
        if self.pre_cluster_weights is not None:
            self.cluster_weights = self.pre_cluster_weights * self.momentum + self.cluster_weights * (1 - self.momentum)
        self.pre_cluster_weights = self.cluster_weights.clone()
        # self.torch_display("cluster_weights ", self.cluster_weights)
        
        return self.cluster_weights


    def torch_display(self, info, tensor):
        tensor = tensor.cpu().tolist()
        rounded_tensor = [round(d, 5) for d in tensor]
        print(info, rounded_tensor)
