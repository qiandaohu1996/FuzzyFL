from typing import List, OrderedDict, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from learners.learner import Learner
from utils.torch_utils import partial_average


class LearnersEnsemble(object):
    """
    Iterable Ensemble of Learners.

    Attributes
    ----------
    learners
    learners_weights
    model_dim
    is_binary_classification
    device
    metric

    Methods
    ----------
    __init__
    __iter__
    __len__
    compute_gradients_and_loss
    optimizer_step
    fit_epochs
    evaluate
    gather_losses
    free_memory
    free_gradients

    """

    def __init__(self, learners, learners_weights):
        self.learners = learners
        self.learners_weights = learners_weights

        self.model_dim = self.learners[0].model_dim
        self.is_binary_classification = self.learners[0].is_binary_classification
        self.device = self.learners[0].device
        self.metric = self.learners[0].metric

    def __setitem__(self, index, value):
        if isinstance(value, Learner):                    # 假设Learner是可接受的类类型
            self.learners[index] = value
        else:
            raise TypeError("Expected a Learner object.")

    def optimizer_step(self):
        """
        perform one optimizer step, requires the gradients to be already computed
        """
        for learner in self.learners:
            learner.optimizer_step()
            
    def compute_losses(self, batch):
        """
        计算与每个学习者的损失。
        :param batch: 一个包含数据和标签的批次
        :return: 每个数据点和每个学习者的损失
        """
        losses = []
        for learner in self.learners:
            loss_vec = learner.compute_loss(batch)
            losses.append(loss_vec)

        return losses
    
    def compute_gradients_and_loss(self, batch, weights=None):
        """
        compute the gradients and loss over one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss

        """
        losses = []
        for learner in self.learners:
            loss = learner.compute_gradients_and_loss(batch, weights=weights)
            losses.append(loss)

        return losses

    def fit_batch(self, batch, weights):

        for learner_id, learner in enumerate(self.learners):
            if weights is not None:
                learner.fit_batch(batch=batch, weights=weights[learner_id])
            else:
                learner.fit_batch(batch=batch, weights=None)

    def fit_batch_record_update(self, batch, weights):

        client_updates = torch.zeros(len(self.learners), self.model_dim)

        for learner_id, learner in enumerate(self.learners):
            old_params = learner.get_param_tensor()
            if weights is not None:
                learner.fit_batch(batch=batch, weights=weights[learner_id])
            else:
                learner.fit_batch(batch=batch, weights=None)
            params = learner.get_param_tensor()
            client_updates[learner_id] = (params - old_params)

        return client_updates.cpu().numpy()

    def fit_batches(self, batch, n_batches, weights):

        for learner_id, learner in enumerate(self.learners):
            if weights is not None:
                learner.fit_batches(batch, n_batches, weights=weights[learner_id])
            else:
                learner.fit_batches(batch, n_batches, weights=None)

    def fit_batches_record_update(self, batch, n_batches, weights):

        client_updates = torch.zeros(len(self.learners), self.model_dim)

        for learner_id, learner in enumerate(self.learners):
            old_params = learner.get_param_tensor()
            if weights is not None:
                learner.fit_batches(batch, n_batches, weights=weights[learner_id])
            else:
                learner.fit_batches(batch, n_batches, weights=None)
            params = learner.get_param_tensor()

            client_updates[learner_id] = (params - old_params)

        return client_updates.cpu().numpy()

    def fit_epoch(self, iterator, weights=None):

        for learner_id, learner in enumerate(self.learners):
            if weights is not None:
                learner.fit_epoch(iterator, weights=weights[learner_id])
            else:
                learner.fit_epoch(iterator, weights=None)

    def fit_epochs(self, iterator, n_epochs, weights=None):

        for learner_id, learner in enumerate(self.learners):
            if weights is not None:
                learner.fit_epochs(iterator, n_epochs, weights=weights[learner_id])
            else:
                learner.fit_epochs(iterator, n_epochs, weights=None)

    def fit_epochs_record_update(self, iterator, n_epochs, weights=None):
        """
        perform multiple training epochs, updating each learner in the ensemble

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param n_epochs: number of epochs
        :type n_epochs: int
        :param weights: tensor of shape (n_learners, len(iterator)), holding the weight of each sample in iterator
                        for each learner ins ensemble_learners
        :type weights: torch.tensor or None
        :return:
            client_updates (np.array, shape=(n_learners, model_dim)): the difference between the old parameter
            and the updated parameters for each learner in the ensemble.

        """
        client_updates = torch.zeros(len(self.learners), self.model_dim)

        for learner_id, learner in enumerate(self.learners):
            old_params = learner.get_param_tensor()
            if weights is not None:
                learner.fit_epochs(iterator, n_epochs, weights=weights[learner_id])
            else:
                learner.fit_epochs(iterator, n_epochs, weights=None)
            params = learner.get_param_tensor()

            client_updates[learner_id] = (params - old_params)

        return client_updates.cpu().numpy()

    def evaluate_iterator(self, iterator):
        """
        Evaluate a ensemble of learners on iterator.

        :param iterator: yields x, y, indices
        :type iterator: torch.utils.data.DataLoader
        :return: global_loss, global_acc

        """
        if self.is_binary_classification:
            criterion = nn.BCELoss(reduction="none")
        else:
            criterion = nn.NLLLoss(reduction="none")

        for learner in self.learners:
            learner.model.eval()
        global_loss = 0.
        global_metric = 0.
        n_samples = 0
        with torch.no_grad():
            for (x, y, _) in iterator:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)
                n_samples += y.size(0)
                y_pred = 0.

                for learner_id, learner in enumerate(self.learners):
                    if self.is_binary_classification:
                        y_pred += self.learners_weights[learner_id] * torch.sigmoid(learner.model(x))
                    else:
                        y_pred += self.learners_weights[learner_id] * F.softmax(learner.model(x), dim=1)

                y_pred = torch.clamp(y_pred, min=0., max=1.)

                if self.is_binary_classification:
                    y = y.type(torch.float32).unsqueeze(1)
                    global_loss += criterion(y_pred, y).sum().item()
                    y_pred = torch.logit(y_pred, eps=1e-10)
                else:
                    global_loss += criterion(torch.log(y_pred), y).sum().item()

                global_metric += self.metric(y_pred, y).item()

            return global_loss / n_samples, global_metric / n_samples

    def gather_losses(self, iterator):
        """
        gathers losses for all sample in iterator for each learner in ensemble

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            tensor (n_learners, n_samples) with losses of all elements of the iterator.dataset

        """
        n_samples = len(iterator.dataset)
        all_losses = torch.zeros(len(self.learners), n_samples)
        for learner_id, learner in enumerate(self.learners):
            all_losses[learner_id] = learner.gather_losses(iterator)

        return all_losses

    def free_memory(self):
        """
        free_memory: free the memory allocated by the model weights
        """
        for learner in self.learners:
            learner.free_memory()

    def free_gradients(self):
        """
        free memory allocated by gradients
        """
        for learner in self.learners:
            learner.free_gradients()

    def __iter__(self):
        return LearnersEnsembleIterator(self)

    def __len__(self):
        return len(self.learners)

    def __getitem__(self, idx):
        return self.learners[idx]


class AGFLLearnersEnsemble(LearnersEnsemble):

    def __init__(self, learners, learners_weights, alpha, adaptive_alpha, lr_lambda=0.05):
        super().__init__(learners=learners, learners_weights=learners_weights)

        self.alpha = torch.tensor(
            [alpha] + [(1 - alpha) / (len(learners) - 1)] * (len(learners) - 1), device=self.device
        )

        self.adaptive_alpha = adaptive_alpha
        self.pre_model = None
        self.gd_model = None

    def update_alpha(self):
        # n_clusters = len(self.learners) - 1
        alpha_grad = torch.zeros(len(self.learners), device=self.device)
        eta_alpha = self.learners[0].get_optimizer_lr()

        for i in range(1, len(self.learners)):
            for local_param, cluster_param in zip(
                trainable_params(self.learners[0].model), trainable_params(self.learners[i].model)
            ):

                diff = (local_param.data - cluster_param.data).view(-1)
                grad = (self.alpha[i] * local_param.grad.data + (1 - self.alpha[i]) * cluster_param.grad.data).view(-1)
                alpha_grad[i] += diff @ grad

            alpha_grad[i] += 0.02 * self.alpha[i]
            self.alpha[i] -= eta_alpha * alpha_grad[i]

        # softmax normalization
        self.alpha = torch.exp(self.alpha) / torch.sum(torch.exp(self.alpha))
        self.alpha = torch.clamp(self.alpha, 0.0001, 0.0999)

    def fit_batch(self, batch, weights):

        client_updates = torch.zeros(len(self.learners), self.model_dim)

        for learner_id, learner in enumerate(self.learners):

            print("learner", learner_id)
            old_params = learner.get_param_tensor()
            if weights is not None:
                learner.fit_batch(batch=batch, weights=weights[learner_id])
            else:
                learner.fit_batch(batch=batch, weights=None)
            params = learner.get_param_tensor()
            client_updates[learner_id] = (params - old_params)

        if self.adaptive_alpha:
            self.update_alpha()
        partial_average([self.learners[0]], self.learners[1], self.alpha)

        return client_updates.cpu().numpy()


class APFLLearnersEnsemble(LearnersEnsemble):

    def __init__(self, learners, learners_weights, alpha, adaptive_alpha):
        super().__init__(learners=learners, learners_weights=learners_weights)

        self.alpha = alpha
        self.adaptive_alpha = adaptive_alpha

    def update_alpha(self):
        alpha_grad = 0
        local_lr = self.learners[0].get_optimizer_lr()
        # for local_param, global_param in zip (self.learners[0].model.parameters(), self.learners[1].model.parameters()) :
        for local_param, global_param in zip(
            trainable_params(self.learners[0].model), trainable_params(self.learners[1].model)
        ):
            diff = (local_param.data - global_param.data).flatten()
            grad = (self.alpha * local_param.grad.data + (1 - self.alpha) * global_param.grad.data).flatten()
            alpha_grad += diff @ grad

        alpha_grad += 0.02 * self.alpha
        self.alpha -= local_lr * alpha_grad
        self.alpha = np.clip(self.alpha.cpu().numpy(), 0.0000001, 0.999999)


class GroupAPFLLearnersEnsemble(LearnersEnsemble):

    def __init__(self, learners, learners_weights, alpha, adaptive_alpha, lr=0.05):
        super().__init__(learners=learners, learners_weights=learners_weights)

        self.alpha = alpha
        self.adaptive_alpha = adaptive_alpha

    def update_alpha(self):
        alpha_grad = 0
        # for local_param, global_param in zip (self.learners[0].model.parameters(), self.learners[1].model.parameters()) :
        for local_param, global_param in zip(
            trainable_params(self.learners[0].model), trainable_params(self.learners[1].model)
        ):
            diff = (local_param.data - global_param.data).flatten()
            grad = (self.alpha * local_param.grad.data + (1 - self.alpha) * global_param.grad.data).flatten()
            alpha_grad += diff @ grad

        alpha_grad += 0.02 * self.alpha
        self.alpha -= self.lr_lambda * alpha_grad
        self.alpha = np.clip(self.alpha.cpu().numpy(), 0.001, 0.999)

    def fit_batch(self, batch, weights):

        client_updates = torch.zeros(len(self.learners), self.model_dim)

        for learner_id, learner in enumerate(self.learners):

            print("learner", learner_id)
            old_params = learner.get_param_tensor()
            if weights is not None:
                learner.fit_batch(batch=batch, weights=weights[learner_id])
            else:
                learner.fit_batch(batch=batch, weights=None)
            params = learner.get_param_tensor()
            client_updates[learner_id] = (params - old_params)

        if self.adaptive_alpha:
            self.update_alpha()
        partial_average([self.learners[0]], self.learners[1], self.alpha)

        return client_updates.cpu().numpy()


class LanguageModelingLearnersEnsemble(LearnersEnsemble):

    def evaluate_iterator(self, iterator):
        """
        Evaluate a ensemble of learners on iterator.

        :param iterator: yields x, y, indices
        :type iterator: torch.utils.data.DataLoader
        :return: global_loss, global_acc

        """
        criterion = nn.NLLLoss(reduction="none")

        for learner in self.learners:
            learner.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for (x, y, _) in iterator:
                x = x.to(self.device)
                y = y.to(self.device)
                n_samples += y.size(0)
                chunk_len = y.size(1)

                y_pred = 0.
                for learner_id, learner in enumerate(self.learners):
                    y_pred += self.learners_weights[learner_id] * F.softmax(learner.model(x), dim=1)

                y_pred = torch.clamp(y_pred, min=0., max=1.)

                global_loss += criterion(torch.log(y_pred), y).sum().item() / chunk_len
                global_metric += self.metric(y_pred, y).item() / chunk_len

            return global_loss / n_samples, global_metric / n_samples


class LearnersEnsembleIterator(object):
    """
    LearnersEnsemble iterator class

    Attributes
    ----------
    _learners_ensemble
    _index

    Methods
    ----------
    __init__
    __next__

    """

    def __init__(self, learners_ensemble):
        self._learners_ensemble = learners_ensemble.learners
        self._index = 0

    def __next__(self):
        while self._index < len(self._learners_ensemble):
            result = self._learners_ensemble[self._index]
            self._index += 1

            return result

        raise StopIteration


def trainable_params(src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module],
                     requires_name=False) -> Union[List[torch.Tensor], Tuple[List[str], List[torch.Tensor]]]:
    parameters = []
    keys = []
    if isinstance(src, OrderedDict):
        for name, param in src.items():
            if param.requires_grad:
                parameters.append(param)
                keys.append(name)
    elif isinstance(src, torch.nn.Module):
        for name, param in src.state_dict(keep_vars=True).items():
            if param.requires_grad:
                parameters.append(param)
                keys.append(name)

    if requires_name:
        return keys, parameters
    else:
        return parameters
    