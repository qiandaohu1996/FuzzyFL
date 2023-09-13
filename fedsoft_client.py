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
        
        self.sigma = 1e-4


    def update_cluster_weights(self,n_clusters):
        """
        Update the cluster weights based on the distance from cluster models (learners)
        """
        data_point_counts = torch.zeros(n_clusters, dtype=torch.float32)
        
        for batch in self.train_iterator:
            x, y, indices = batch
            batch_losses = self.learners_ensemble.compute_losses(batch)
            for j in range(len(indices)):
                min_loss_idx = torch.argmin(torch.tensor([loss[j] for loss in batch_losses]))
                data_point_counts[min_loss_idx] += 1

        # 归一化
        data_point_counts /= data_point_counts.sum()
        # 使用 torch.clamp 进行剪切，确保每个权重不小于 self.sigma
        cluster_weights = torch.clamp(data_point_counts, min=self.sigma)
        # 再次归一化以确保剪切后的权重之和为1
        cluster_weights /= cluster_weights.sum()

        return cluster_weights

