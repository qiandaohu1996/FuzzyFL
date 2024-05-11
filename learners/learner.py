import torch

class Learner:
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
        is_byzantine=False,  # 是否为拜占庭学习器
        z_max=0  # 拜占庭攻击强度
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.metric = metric
        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.fuzzy_m_scheduler = fuzzy_m_scheduler
        self.is_binary_classification = is_binary_classification
        self.cluster_models=None
        self.cluster_weights=None
        self.model_dim = int(self.get_param_tensor().shape[0])
        self.is_byzantine = is_byzantine
        self.z_max = z_max   

    def optimizer_step(self):
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def get_optimizer_lr(self):
        learning_rate = self.optimizer.param_groups[0]["lr"]
        return learning_rate
    
    def alittle_byzantine_attack(self):
        """应用拜占庭攻击到模型的梯度上"""
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    std = param.grad.std()
                    param.grad += self.z_max * std

    def fit_batch(self, batch, weights=None):
        self.model.train()
        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)
        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)

        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)
        # loss = loss_vec.mean() if weights is None else (loss_vec * weights[indices]).sum() / loss_vec.size(0)
        
        if weights is not None:
            weights = weights.to(self.device) 
            loss = (torch.sum(loss_vec * weights[indices])) / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        loss.backward()
        
        if self.is_byzantine:
            self.alittle_byzantine_attack()   
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

        return loss.detach(), self.metric(y_pred, y).detach() / len(y)

    
    def compute_gradients_and_loss(self, batch, weights=None):
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
            loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
        else:
            loss = loss_vec.mean()
        loss.backward()
        return loss.detach()
    
    def fit_batches(self, batches, weights=None):
        for batch in batches:
            self.fit_batch(batch, weights)
            
    def fit_epoch(self, iterator, weights=None):
        self.model.train()
        total_loss = 0
        total_metric = 0
        num_batches = 0

        for batch in iterator:
            loss, metric = self.fit_batch(batch, weights)
            total_loss += loss
            total_metric += metric
            num_batches += 1
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.0)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_metric = total_metric / num_batches if num_batches > 0 else 0

        return avg_loss, avg_metric
         
    def fit_epoch(self, iterator, weights=None): 
        self.model.train() 
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
                loss = (loss_vec * weights[indices]).sum() / loss_vec.size(0)
            else:
                loss = loss_vec.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.0)
            # gradient_norm = torch.norm(torch.cat([param.grad.flatten() for param in self.model.parameters()]))
            self.optimizer.step()

    def fit_epochs(self, iterator, n_epochs, weights=None):
        for _ in range(n_epochs):
            self.fit_epoch(iterator, weights)
            
    def fit_batch_record_update(self, batch, weights=None):
        client_updates = torch.zeros(self.model_dim)
        old_params = self.get_param_tensor()
        self.fit_batch(batch, weights)
        params = self.get_param_tensor()
        client_updates = params - old_params
        return client_updates.cpu().numpy()

    def fit_batches_record_update(self, batches, weights=None):
        client_updates = torch.zeros(self.model_dim, device=self.device)
        old_params = self.get_param_tensor()
        for batch in batches:
            self.fit_batch(batch, weights)
            params = self.get_param_tensor()
            client_updates += params - old_params
        return client_updates.cpu().numpy()
    
    def fit_epochs_record_update(self, iterator, n_epochs, weights=None):
        client_updates = torch.zeros(self.model_dim)
        old_params = self.get_param_tensor()
        self.fit_epochs(iterator, n_epochs, weights)
        params = self.get_param_tensor()
        client_updates = params - old_params
        return client_updates.cpu().numpy()

    def gather_losses(self, iterator):
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
        self.model.eval()

        global_loss = 0.0
        global_metric = 0.0
        n_samples = 0

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
        param_list = []
        for param in self.model.parameters():
            param_list.append(param.data.view(-1,))
        return torch.cat(param_list)

    def get_grad_tensor(self): 
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
 
class Learner1:
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
        is_byzantine=False,  # 是否为拜占庭学习器
        z_max=None  # 拜占庭攻击强度
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.metric = metric
        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.fuzzy_m_scheduler = fuzzy_m_scheduler
        self.is_binary_classification = is_binary_classification
        self.is_byzantine = is_byzantine
        self.z_max = z_max or 0  # 设置默认攻击强度为0

    def apply_byzantine_attack(self):
        """应用拜占庭攻击到模型的梯度上"""
        if self.is_byzantine:
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        std = param.grad.std()
                        param.grad += self.z_max * std

    def fit_batch(self, batch, weights=None):
        self.model.train()
        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)
        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)

        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)
        loss = loss_vec.mean() if weights is None else (loss_vec * weights[indices]).sum() / loss_vec.size(0)
        loss.backward()

        self.apply_byzantine_attack()  # 在步骤后调用拜占庭攻击

        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

        return loss.detach(), self.metric(y_pred, y).detach() / len(y)

    def fit_epoch(self, iterator, weights=None):
        self.model.train()
        total_loss = 0
        total_metric = 0
        num_batches = 0

        for batch in iterator:
            loss, metric = self.fit_batch(batch, weights)
            total_loss += loss
            total_metric += metric
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_metric = total_metric / num_batches if num_batches > 0 else 0

        return avg_loss, avg_metric

        
class ByzantineLearner(Learner):
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
        z_max=None  # 拜占庭攻击强度
    ):
        super().__init__(
            model,
            criterion,
            metric,
            device,
            optimizer,
            lr_scheduler,
            fuzzy_m_scheduler,
            is_binary_classification,
            is_byzantine=True,  # 默认标记为拜占庭学习器
            z_max=z_max
        )

    def fit_batch(self, batch, weights=None):
        self.model.train()
        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)
        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)

        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)
        loss = loss_vec.mean() if weights is None else (loss_vec * weights[indices]).sum() / loss_vec.size(0)
        loss.backward()

        if self.is_byzantine:
            with torch.no_grad():
                for param in self.model.parameters():
                    std = param.grad.std()
                    param.grad += self.z_max * std

        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

        return loss.detach(), self.metric(y_pred, y).detach() / len(y)

    def fit_epoch(self, iterator, weights=None):
        self.model.train()
        total_loss = 0
        total_metric = 0
        num_batches = 0

        for batch in iterator:
            loss, metric = self.fit_batch(batch, weights)
            total_loss += loss
            total_metric += metric
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_metric = total_metric / num_batches if num_batches > 0 else 0

        return avg_loss, avg_metric



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
