import torch

# from utils.my_profiler import *


class Learner:
    """
    Responsible of training and evaluating a (deep-)learning model

    Attributes
    ----------
    model (nn.Module): the model trained by the learner

    criterion (torch.nn.modules.loss): loss function used to train the `model`, should have reduction="none"

    metric (fn): function to compute the metric, should accept as input two vectors and return a scalar

    device (str or torch.device):

    optimizer (torch.optim.Optimizer):

    lr_scheduler (torch.optim.lr_scheduler):

    is_binary_classification (bool): whether to cast labels to float or not, if `BCELoss`
    is used as criterion this should be set to True

    Methods
    ------
    compute_gradients_and_loss:

    optimizer_step: perform one optimizer step, requires the gradients to be already computed.

    fit_batch: perform an optimizer step over one batch

    fit_epoch:

    fit_batches: perform successive optimizer steps over successive batches

    fit_epochs:

    evaluate_iterator: evaluate `model` on an iterator

    gather_losses:

    get_param_tensor: get `model` parameters as a unique flattened tensor

    free_memory: free the memory allocated by the model weights

    free_gradients:
    """

    def __init__(
        self,
        model,
        criterion,
        metric,
        device,
        optimizer,
        lr_scheduler=None,
        fuzzy_m_scheduler=None,
        is_binary_classification=False,
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.metric = metric
        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.fuzzy_m_scheduler = fuzzy_m_scheduler
        self.is_binary_classification = is_binary_classification

        self.model_dim = int(self.get_param_tensor().shape[0])

    def optimizer_step(self):
        """
        perform one optimizer step, requires the gradients to be already computed.
        """
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def get_optimizer_lr(self):
        learning_rate = self.optimizer.param_groups[0]["lr"]
        return learning_rate

    def compute_gradients_and_loss(self, batch, weights=None):
        """
        compute the gradients and loss over one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss gradient

        """
        self.model.train()

        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)
        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)

        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)
        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec * weights[indices]).sum() / loss_vec.size(0)
        else:
            loss = loss_vec.mean()
        loss.backward()

        return loss.detach()

    def fit_batch(self, batch, weights=None):
        self.model.train()
        self.optimizer.zero_grad()
        self.compute_gradients_and_loss(batch, weights)

        # gradient_norm = torch.norm(torch.cat([param.grad.flatten() for param in self.model.parameters()]))
        # print(f"before Gradient Norm: {gradient_norm.item():.3f}",)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.0)
        self.optimizer.step()

    def fit_batch_record_update(self, batch, weights=None):
        client_updates = torch.zeros(self.model_dim)
        old_params = self.get_param_tensor()
        self.fit_batch(batch, weights)
        params = self.get_param_tensor()
        client_updates = params - old_params

        return client_updates.cpu().numpy()

    def fit_batches(self, batches, weights=None):
        for batch in batches:
            self.fit_batch(batch, weights)

    def fit_batches_record_update(self, batches, weights=None):
        client_updates = torch.zeros(self.model_dim, device=self.device)
        old_params = self.get_param_tensor()

        for batch in batches:
            self.fit_batch(batch, weights)
            params = self.get_param_tensor()
            client_updates += params - old_params

        return client_updates.cpu().numpy()

    # @memory_profiler
    def fit_epoch(self, iterator, weights=None):
        # client_updates = torch.zeros(self.model_dim)
        # old_params = self.get_param_tensor()
        # print("old_params" ,old_params[150:155])

        self.model.train()

        # global_loss = 0.
        # global_metric = 0.
        n_samples = 0

        for x, y, indices in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)
            n_samples += y.size(0)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)

            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss_vec = self.criterion(y_pred, y)
            if weights is not None:
                weights = weights.to(self.device)
                # loss_vec=loss_vec.unsqueeze(0)

                # print("loss_vec size",loss_vec.size())
                # print("weights[indices] size",weights[indices].size())
                loss = (loss_vec * weights[indices]).sum() / loss_vec.size(0)
            else:
                loss = loss_vec.mean()
            loss.backward()

            # gradient_norm = torch.norm(torch.cat([param.grad.flatten() for param in self.model.parameters()]))

            # 输出梯度范数的值
            # print(f"Gradient Norm:{gradient_norm.item():.3f}",)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.0)
            # gradient_norm = torch.norm(torch.cat([param.grad.flatten() for param in self.model.parameters()]))

            # 输出梯度范数的值
            # print(f"Gradient Norm:{gradient_norm.item():.3f}",)

            self.optimizer.step()
            # params = self.get_param_tensor()
            # client_updates = (params - old_params)
            # print(params[150:155])
            # print(client_updates[150:155])

        #     global_loss += loss.detach() * loss_vec.size(0)
        #     global_metric += self.metric(y_pred, y).detach()
        #     # print("loss ", global_loss / n_samples)

        # return global_loss / n_samples, global_metric / n_samples

    def fit_epochs(self, iterator, n_epochs, weights=None):
        for _ in range(n_epochs):
            self.fit_epoch(iterator, weights)

    def fit_epochs_record_update(self, iterator, n_epochs, weights=None):
        client_updates = torch.zeros(self.model_dim)
        old_params = self.get_param_tensor()
        self.fit_epochs(iterator, n_epochs, weights)
        params = self.get_param_tensor()
        client_updates = params - old_params
        return client_updates.cpu().numpy()

    def saga_fit_epoch(self, iterator, weights=None):
        self.model.train()
        client_updates = torch.zeros(self.model_dim)

        n_samples = 0
        old_params = self.get_param_tensor()

        for x, y, indices in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            n_samples += y.size(0)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)

            self.optimizer.zero_grad()

            y_pred = self.model(x)

            loss_vec = self.criterion(y_pred, y)
            if weights is not None:
                weights = weights.to(self.device)
                loss = (loss_vec * weights[indices]).sum() / loss_vec.size(0)
                # loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)

            else:
                loss = loss_vec.mean()
            loss.backward()
            # client_grad=self.model.grad
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.0)

            self.optimizer.step()
            params = self.get_param_tensor()

            client_updates = params - old_params

        return client_updates.cpu().numpy()

    def gather_losses(self, iterator):
        """
        gathers losses for all elements of iterator

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            tensor with losses of all elements of the iterator.dataset

        """
        self.model.eval()
        n_samples = len(iterator.dataset)
        all_losses = torch.zeros(n_samples, device=self.device)

        with torch.no_grad():
            for x, y, indices in iterator:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)

                if self.is_binary_classification:
                    y = y.type(torch.float32).unsqueeze(1)

                y_pred = self.model(x)
                all_losses[indices] = self.criterion(y_pred, y).squeeze()

        return all_losses

    def evaluate_iterator(self, iterator):
        """
        evaluate learner on `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            global_loss and  global_metric accumulated over the iterator

        """
        self.model.eval()

        global_loss = 0.0
        global_metric = 0.0
        n_samples = 0
        # len_iter=more_itertools.ilen(iterator)
        # if len_iter<64:

        for x, y, _ in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)
            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)

            with torch.no_grad():
                y_pred = self.model(x)

                global_loss += self.criterion(y_pred, y).sum().detach()
                global_metric += self.metric(y_pred, y).detach()

            n_samples += y.size(0)

        return global_loss / n_samples, global_metric / n_samples

    def get_param_tensor(self):
        """
        get `model` parameters as a unique flattened tensor

        :return: torch.tensor

        """
        param_list = []

        for param in self.model.parameters():
            param_list.append(param.data.view(-1,))

        return torch.cat(param_list)

    def get_grad_tensor(self):
        """
        get `model` gradients as a unique flattened tensor
        :return: torch.tensor
        """
        grad_list = []

        for param in self.model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.data.view(-1,))

        return torch.cat(grad_list)

    def free_memory(self):
        del self.optimizer
        del self.model

    def free_gradients(self):
        self.optimizer.zero_grad(set_to_none=True)


class LanguageModelingLearner(Learner):
    def fit_batch(self, batch, weights=None):
        self.model.train()

        x, y, indices = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # n_samples = y.size(0)
        # chunk_len = y.size(1)
        self.optimizer.zero_grad()

        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]).mean() / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        loss.backward()

        self.optimizer.step()

        # global_loss = loss.detach() * loss_vec.size(0) / chunk_len
        # global_metric = self.metric(y_pred, y).detach() / chunk_len

        # return global_loss / n_samples, global_metric / n_samples

    def fit_batch_record_update(self, batch, weights=None):
        client_updates = torch.zeros(self.model_dim)
        old_params = self.get_param_tensor()
        self.fit_batch(batch, weights)
        params = self.get_param_tensor()
        client_updates = params - old_params

        return client_updates.cpu().numpy()

    def fit_batches(self, batches, weights=None):
        for batch in batches:
            self.fit_batch(batch, weights)

    def fit_batches_record_update(self, batches, weights=None):
        client_updates = torch.zeros(self.model_dim, device=self.device)
        old_params = self.get_param_tensor()

        for batch in batches:
            self.fit_batch(batch, weights)
            params = self.get_param_tensor()
            client_updates += params - old_params

        return client_updates.cpu().numpy()

    def fit_epoch(self, iterator, weights=None):
        self.model.train()

        # global_loss = 0.0
        # global_metric = 0.0
        n_samples = 0

        for x, y, indices in iterator:
            x = x.to(self.device)
            y = y.to(self.device)

            n_samples += y.size(0)
            # chunk_len = y.size(1)
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss_vec = self.criterion(y_pred, y)

            if weights is not None:
                weights = weights.to(self.device)
                loss = (loss_vec.T @ weights[indices]).mean() / loss_vec.size(0)
            else:
                loss = loss_vec.mean()

            loss.backward()
            self.optimizer.step()
            # global_loss += loss.detach() * loss_vec.size(0) / chunk_len
            # global_metric += self.metric(y_pred, y).detach() / chunk_len

        # return global_loss / n_samples, global_metric / n_samples

    def fit_epochs_record_update(self, iterator, n_epochs, weights=None):
        client_updates = torch.zeros(self.model_dim)
        old_params = self.get_param_tensor()
        self.fit_epochs(iterator, n_epochs, weights)
        params = self.get_param_tensor()
        client_updates = params - old_params
        return client_updates.cpu().numpy()

    @torch.no_grad()
    def gather_losses(self, iterator):
        """
        gathers losses for all elements of iterator

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            tensor with losses of all elements of the iterator.dataset

        """
        self.model.eval()
        n_samples = len(iterator.dataset)
        predictions = torch.zeros(n_samples, device=self.device)

        for x, y, indices in iterator:
            x = x.to(self.device)
            y = y.to(self.device)

            y_pred = self.model(x)
            predictions[indices] = self.criterion(y_pred, y).mean(axis=1)

        return predictions

    @torch.no_grad()
    def evaluate_iterator(self, iterator):
        """
        evaluate learner on `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            global_loss and  global_metric accumulated over the iterator

        """
        self.model.eval()

        global_loss = 0.0
        global_metric = 0.0
        n_samples = 0

        for x, y, _ in iterator:
            x = x.to(self.device)
            y = y.to(self.device)
            n_samples += y.size(0)

            chunk_len = y.size(1)

            y_pred = self.model(x)
            global_loss += self.criterion(y_pred, y).sum().detach() / chunk_len
            global_metric += self.metric(y_pred, y).detach() / chunk_len

        return global_loss / n_samples, global_metric / n_samples
