import os
import numpy as np
import torch.nn.functional as F
import torch
from copy import deepcopy
from utils.torch_utils import *
from utils.constants import *

from utils.datastore import *

import warnings


class Client(object):
    r"""Implements one clients

    Attributes
    ----------
    learners_ensemble
    n_learners

    train_iterator

    val_iterator

    test_iterator

    train_loader

    n_train_samples

    n_test_samples

    samples_weights

    local_steps

    logger

    tune_locally:

    Methods
    ----------
    __init__
    step
    write_logs
    update_sample_weights
    update_learners_weights

    """

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
        self.learners_ensemble = learners_ensemble

        self.n_learners = len(self.learners_ensemble)
        self.tune_locally = tune_locally

        if self.tune_locally:
            self.tuned_learners_ensemble = deepcopy(self.learners_ensemble)
        else:
            self.tuned_learners_ensemble = None

        self.binary_classification_flag = (
            self.learners_ensemble.is_binary_classification
        )

        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.test_iterator = test_iterator

        self.train_loader = iter(self.train_iterator)

        # self.preload_num = 4
        # self.batch_queue = queue.Queue(maxsize=self.preload_num)
        # self.stop_event = threading.Event()
        # self.thread = threading.Thread(target=self._preload)
        # self.thread.start()

        self.n_train_samples = len(self.train_iterator.dataset)
        self.n_test_samples = len(self.test_iterator.dataset)

        self.samples_weights = (
            torch.ones(self.n_learners, self.n_train_samples) / self.n_learners
        )

        self.local_steps = local_steps
        self.idx = idx
        self.counter = 0
        self.logger = logger

    def _preload(self):
        while not self.stop_event.is_set():
            try:
                batch = next(self.train_loader)
                self.batch_queue.put(batch)
            except StopIteration:
                self.train_loader = iter(self.train_iterator)
            except Exception as e:
                print(f"Exception in get_next_batch: {e}")

    def get_next_batch(self):
        try:
            batch = next(self.train_loader)
        except StopIteration:
            # print(f"StopIteration in get_next_batch")
            self.train_loader = iter(self.train_iterator)
            batch = next(self.train_loader)
        except Exception as e:
            print(f"Exception in get_next_batch: {e}")
            batch = None
        return batch

    def step(self, single_batch_flag=True, *args, **kwargs):
        """
        perform on step for the client

        :param single_batch_flag: if true, the client only uses one batch to perform the update
        :return
            clients_updates: ()
        """
        self.counter += 1
        self.update_sample_weights()
        self.update_learners_weights()

        if single_batch_flag:
            for _ in range(self.local_steps):
                batch = self.get_next_batch()
                # client_updates = \
                self.learners_ensemble.fit_batch(
                    batch=batch, weights=self.samples_weights
                )
        else:
            # client_updates = \
            self.learners_ensemble.fit_epochs(
                iterator=self.train_iterator,
                n_epochs=self.local_steps,
                weights=self.samples_weights,
            )
        for learner in self.learners_ensemble:
            if learner.lr_scheduler:
                learner.lr_scheduler.step()

        # TODO: add flag arguments to use `free_gradients`
        # self.learners_ensemble.free_gradients()

        # return client_updates

    def step_record_update(self, single_batch_flag=True, *args, **kwargs):
        """
        perform on step for the client

        :param single_batch_flag: if true, the client only uses one batch to perform the update
        :return
            clients_updates: ()
        """
        self.counter += 1
        # self.update_sample_weights()
        # self.update_learners_weights()
        # print("samples_weights size",self.samples_weights.size())
        if single_batch_flag:
            batches = []
            for _ in range(self.local_steps):
                batch = self.get_next_batch()
                batches.append(batch)

            client_updates = self.learners_ensemble[0].fit_batches_record_update(
                batches=batches, weights=self.samples_weights[0] *
                self.n_learners
            )
            # print("\nclient_updates ", client_updates)

        else:
            client_updates = self.learners_ensemble[0].fit_epochs_record_update(
                iterator=self.train_iterator,
                n_epochs=self.local_steps,
                weights=self.samples_weights[0] * self.n_learners,
            )

        # TODO: add flag arguments to use `free_gradients`
        # self.learners_ensemble.free_gradients()

        return client_updates

    def _write_logs(self, logger, iterator, log_type):
        if self.tune_locally:
            self.update_tuned_learners()

        learners_ensemble = (
            self.tuned_learners_ensemble
            if self.tune_locally
            else self.learners_ensemble
        )
        # batch = self.get_next_batch()

        loss, acc = learners_ensemble.evaluate_iterator(iterator)

        logger.add_scalar(f"{log_type}/Loss", loss, self.counter)
        logger.add_scalar(f"{log_type}/Metric", acc, self.counter)
        return loss, acc

    def write_train_logs(self):
        return self._write_logs(self.logger, self.val_iterator, "Train")

    def write_test_logs(self):
        return self._write_logs(self.logger, self.test_iterator, "Test")

    def write_local_test_logs(self):
        return self._write_logs(self.logger, self.test_iterator, "Local_Test")

    def write_logs(self):
        train_loss, train_acc = self.write_train_logs()
        test_loss, test_acc = self.write_test_logs()

        return train_loss, train_acc, test_loss, test_acc

    def update_sample_weights(self):
        pass

    def update_learners_weights(self):
        pass

    def update_tuned_learners(self):
        if not self.tune_locally:
            return

        for learner_id, learner in enumerate(self.tuned_learners_ensemble):
            copy_model(
                source=self.learners_ensemble[learner_id].model, target=learner.model
            )
            learner.fit_epochs(
                self.train_iterator,
                self.local_steps,
                weights=self.samples_weights[learner_id],
            )

    def save_state(self, client_id, dir_path):
        """
        save the state of the client, i.e., the state dictionary of each `learner` in `learners_ensemble`
        as `.pt` file, and `learners_weights` as a single numpy array (`.np` file).

        :param dir_path:
        """
        for learner_id, learner in enumerate(self.learners_ensemble):
            save_path = os.path.join(
                dir_path, f"chkpts_client{client_id}_learner{learner_id}.pt"
            )
            torch.save(learner.model.state_dict(), save_path)

        # save learners_weights
        learners_weights_path = os.path.join(
            dir_path, f"client{client_id}_learners_weights.npy"
        )
        np.save(learners_weights_path, self.learners_ensemble.learners_weights)

    def load_state(self, client_id, dir_path):
        """
        load the state of the client, i.e., the state dictionary of each `learner` in `learners_ensemble`
        from a `.pt` file, and `learners_weights` from numpy array (`.np` file).

        :param dir_path:
        """
        for learner_id, learner in enumerate(self.learners_ensemble):
            chkpts_path = os.path.join(
                dir_path, f"chkpts_client{client_id}_learner{learner_id}.pt"
            )
            learner.model.load_state_dict(torch.load(chkpts_path))

        # load learners_weights
        learners_weights_path = os.path.join(
            dir_path, f"client{client_id}_learners_weights.npy"
        )
        self.learners_ensemble.learners_weights = np.load(
            learners_weights_path)


class AGFLClient(Client):
    def __init__(
        self,
        learners_ensemble,
        train_iterator,
        val_iterator,
        test_iterator,
        logger,
        local_steps,
        tune_locally=False,
    ):
        super(AGFLClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
        )
        self.counter = 0


class FedRepClient(Client):
    """
    Client used to implement
        "Exploiting Shared Representations for Personalized FederatedLearning"__(https://arxiv.org/pdf/2102.07078.pdf)

    """

    def __init__(
        self,
        learner,
        train_iterator,
        val_iterator,
        test_iterator,
        logger,
        local_steps,
        save_path=None,
        id_=None,
        *args,
        **kwargs,
    ):
        super(FedRepClient, self).__init__(
            learner=learner,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            save_path=save_path,
            id_=id_,
            *args,
            **kwargs,
        )
        self.head = deepcopy(self.learner.get_head())

    def step(self):
        head = self.learner.get_head()
        copy_model(source=self.head, target=head)

        self.learner.freeze_body()

        # train the head
        self.learner.fit_epochs(
            iterator=self.train_iterator,
            n_epochs=LOCAL_HEAD_UPDATES,
        )
        self.head = deepcopy(self.learner.get_head())

        # train the body with fixed head
        self.learner.unfreeze_body()

        head = self.learner.get_head()
        # client_updates = \
        self.learner.fit_epochs(
            iterator=self.train_iterator,
            n_epochs=self.local_steps,
            frozen_modules=[head],
        )


class MixtureClient(Client):
    def update_sample_weights(self):
        all_losses = self.learners_ensemble.gather_losses(self.val_iterator)
        self.samples_weights = F.softmax((torch.log(self.learners_ensemble.learners_weights) - all_losses.T), dim=1).T
        torch.cuda.empty_cache()

    def update_learners_weights(self):
        self.learners_ensemble.learners_weights = self.samples_weights.mean(dim=1)
        torch.cuda.empty_cache()


class AgnosticFLClient(Client):
    def __init__(
        self,
        learners_ensemble,
        train_iterator,
        val_iterator,
        test_iterator,
        logger,
        local_steps,
        tune_locally=False,
    ):
        super(AgnosticFLClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
        )

        assert self.n_learners == 1, "AgnosticFLClient only supports single learner."

    def step(self, *args, **kwargs):
        self.counter += 1

        batch = self.get_next_batch()
        losses = self.learners_ensemble.compute_gradients_and_loss(batch)

        return losses


class FFLClient(Client):
    r"""
    Implements client for q-FedAvg from
     `FAIR RESOURCE ALLOCATION IN FEDERATED LEARNING`__(https://arxiv.org/pdf/1905.10497.pdf)

    """

    def __init__(
        self,
        learners_ensemble,
        train_iterator,
        val_iterator,
        test_iterator,
        logger,
        local_steps,
        q=1,
        tune_locally=False,
    ):
        super(FFLClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
        )

        assert self.n_learners == 1, "AgnosticFLClient only supports single learner."
        self.q = q

    def step(self, lr, *args, **kwargs):
        hs = 0
        for learner in self.learners_ensemble:
            initial_state_dict = self.learners_ensemble[0].model.state_dict()
            learner.fit_epochs(iterator=self.train_iterator,
                               n_epochs=self.local_steps)

            client_loss, _ = learner.evaluate_iterator(self.train_iterator)
            client_loss = torch.tensor(client_loss)
            client_loss += 1e-10

            # assign the difference to param.grad for each param in learner.parameters()
            differentiate_learner(
                target=learner,
                reference_state_dict=initial_state_dict,
                coeff=torch.pow(client_loss, self.q) / lr,
            )

            hs = (
                self.q
                * torch.pow(client_loss, self.q - 1)
                * torch.pow(torch.linalg.norm(learner.get_grad_tensor()), 2)
            )
            hs /= torch.pow(torch.pow(client_loss, self.q), 2)
            hs += torch.pow(client_loss, self.q) / lr

        return hs / len(self.learners_ensemble)


class KNNPerClient(Client):
    """

    Attributes
    ----------
    model:
    features_dimension:
    num_classes:
    train_loader:
    test_loader:
    n_train_samples:
    n_test_samples:
    local_steps:
    logger:
    binary_classification_flag:
    counter:
    capacity: datastore capacity of the client
    strategy: strategy to select samples to keep on the datastore
    rng (numpy.random._generator.Generator):
    datastore (datastore.DataStore):
    datastore_flag (bool):
    features_dimension (int):
    num_classes (int):
    train_features: (n_train_samples x features_dimension)
    test_features: (n_train_samples x features_dimension)
    features_flag (bool):
    model_outputs: (n_test_samples x num_classes)
    model_outputs_flag (bool):
    knn_outputs:
    knn_outputs_flag (bool)
    interpolate_logits (bool): if selected logits are interpolated instead of probabilities

    Methods
    -------
    __init__

    build

    compute_features_and_model_outputs

    build_datastore

    gather_knn_outputs

    evaluate

    clear_datastore

    """

    def __init__(
        self,
        learner,
        train_iterator,
        val_iterator,
        test_iterator,
        logger,
        local_steps,
        k,
        interpolate_logits,
        features_dimension,
        num_classes,
        capacity,
        strategy,
        rng,
        *args,
        **kwargs,
    ):
        """

        :param learner:
        :param train_iterator:
        :param val_iterator:
        :param test_iterator:
        :param logger:
        :param local_steps:
        :param k:
        :param features_dimension:
        :param num_classes:
        :param capacity:
        :param strategy:
        :param rng:

        """
        super(KNNPerClient, self).__init__(
            learner=learner,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            *args,
            **kwargs,
        )

        self.k = k
        self.interpolate_logits = interpolate_logits

        self.model = self.learner.model
        self.features_dimension = features_dimension
        self.num_classes = num_classes

        self.train_iterator = train_iterator
        self.test_iterator = test_iterator

        self.n_train_samples = len(train_iterator.dataset)
        self.n_test_samples = len(test_iterator.dataset)

        self.capacity = capacity
        self.strategy = strategy
        self.rng = rng
        self.device = self.learner.device

        self.model = self.model.to(self.device)
        self.model.eval()

        self.datastore = DataStore(
            self.capacity, self.strategy, self.features_dimension, self.rng
        )
        self.datastore_flag = False

        self.train_features = np.zeros(
            shape=(self.n_train_samples,
                   self.features_dimension), dtype=np.float32
        )
        self.train_labels = np.zeros(
            shape=self.n_train_samples, dtype=np.float32)
        self.test_features = np.zeros(
            shape=(self.n_test_samples, self.features_dimension), dtype=np.float32
        )
        self.test_labels = np.zeros(
            shape=self.n_test_samples, dtype=np.float32)
        self.features_flag = False

        self.train_model_outputs = np.zeros(
            shape=(self.n_train_samples, self.num_classes), dtype=np.float32
        )
        self.train_model_outputs_flag = False

        self.test_model_outputs = np.zeros(
            shape=(self.n_test_samples, self.num_classes), dtype=np.float32
        )
        self.test_model_outputs_flag = False

        self.train_knn_outputs = np.zeros(
            shape=(self.n_train_samples, self.num_classes), dtype=np.float32
        )
        self.train_knn_outputs_flag = False

        self.test_knn_outputs = np.zeros(
            shape=(self.n_test_samples, self.num_classes), dtype=np.float32
        )
        self.test_knn_outputs_flag = False

    @property
    def k(self):
        return self.__k

    @k.setter
    def k(self, k):
        self.__k = int(k)

    @property
    def capacity(self):
        return self.__capacity

    @capacity.setter
    def capacity(self, capacity):
        if 0 <= capacity <= 1 and isinstance(capacity, float):
            capacity = int(capacity * self.n_train_samples)
        else:
            capacity = int(capacity)

        if capacity < 0:
            capacity = self.n_train_samples

        self.__capacity = capacity

    def step(self):
        pass

    def compute_features_and_model_outputs(self):
        """
        extract features from `train_iterator` and `test_iterator` .
        and computes the predictions of the base model (i.e., `self.model`) over `test_iterator`.

        """
        self.features_flag = True
        self.train_model_outputs_flag = True
        self.test_model_outputs_flag = True

        (
            self.train_features,
            self.train_model_outputs,
            self.train_labels,
        ) = self.learner.compute_embeddings_and_outputs(
            iterator=self.train_iterator,
            embedding_dim=self.features_dimension,
            n_classes=self.num_classes,
            apply_softmax=(not self.interpolate_logits),
        )

        (
            self.test_features,
            self.test_model_outputs,
            self.test_labels,
        ) = self.learner.compute_embeddings_and_outputs(
            iterator=self.test_iterator,
            embedding_dim=self.features_dimension,
            n_classes=self.num_classes,
            apply_softmax=(not self.interpolate_logits),
        )

    def build_datastore(self):
        assert (
            self.features_flag
        ), "Features should be computed before building datastore!"
        self.datastore_flag = True

        self.datastore.build(self.train_features, self.train_labels)

    def gather_knn_outputs(self, mode="test", scale=1.0):
        """
        computes the k-NN predictions

        :param mode: possible are "train" and "test", default is "test"
        :param scale: scale of the gaussian kernel, default is 1.0
        """
        if self.capacity <= 0:
            warnings.warn(
                "trying to gather knn outputs with empty datastore", RuntimeWarning
            )
            return

        assert (
            self.features_flag
        ), "Features should be computed before building datastore!"
        assert (
            self.datastore_flag
        ), "Should build datastore before computing knn outputs!"

        if mode == "train":
            features = self.train_features
            self.train_knn_outputs_flag = True
        else:
            features = self.test_features
            self.test_knn_outputs_flag = True

        distances, indices = self.datastore.index.search(features, self.k)
        similarities = np.exp(-distances / (self.features_dimension * scale))
        neighbors_labels = self.datastore.labels[indices]

        masks = np.zeros(((self.num_classes,) + similarities.shape))
        for class_id in range(self.num_classes):
            masks[class_id] = neighbors_labels == class_id

        outputs = (similarities * masks).sum(axis=2) / similarities.sum(axis=1)

        if mode == "train":
            self.train_knn_outputs = outputs.T
        else:
            self.test_knn_outputs = outputs.T

    def evaluate(self, weight, mode="test"):
        """
        evaluates the client for a given weight parameter

        :param weight: float in [0, 1]
        :param mode: possible are "train" and "test", default is "test"

        :return:
            accuracy score

        """
        if mode == "train":
            flag = self.train_knn_outputs_flag
            knn_outputs = self.train_knn_outputs
            model_outputs = self.train_model_outputs
            labels = self.train_labels

        else:
            flag = self.test_knn_outputs_flag
            knn_outputs = self.test_knn_outputs
            model_outputs = self.test_model_outputs
            labels = self.test_labels

        if flag:
            outputs = weight * knn_outputs + (1 - weight) * model_outputs
        else:
            warnings.warn(
                "evaluation is done only with model outputs, datastore is empty",
                RuntimeWarning,
            )
            outputs = model_outputs

        predictions = np.argmax(outputs, axis=1)

        correct = (labels == predictions).sum()
        total = len(labels)

        if total == 0:
            acc = 1
        else:
            acc = correct / total

        return acc

    def clear_datastore(self):
        """
        clears `datastore`
        """
        self.datastore.clear()
        self.datastore.capacity = self.capacity

        self.datastore_flag = False
        self.train_knn_outputs_flag = False
        self.test_knn_outputs_flag = False
