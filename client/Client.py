import os
import random
import time
import numpy as np
import torch.nn.functional as F
import torch
from copy import deepcopy
from utils.torch_utils import *
from utils.constants import *

import warnings


class Client(object):

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

        self.binary_classification_flag = self.learners_ensemble.is_binary_classification

        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.test_iterator = test_iterator

        self.train_loader = iter(self.train_iterator)

        self.n_train_samples = len(self.train_iterator.dataset)
        self.n_test_samples = len(self.test_iterator.dataset)

        self.samples_weights = (
            torch.ones(self.n_learners, self.n_train_samples) / self.n_learners
        )

        self.local_steps = local_steps
        self.idx = idx
        self.counter = 0
        
        self.logger = logger

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
      
        self.counter += 1
        # self.update_sample_weights()
        # self.update_learners_weights()
        # print('client counter', self.counter )
        if single_batch_flag:
            for _ in range(self.local_steps):
                batch = self.get_next_batch()
                self.learners_ensemble.fit_batch(
                    batch=batch, weights=self.samples_weights
                )
        else:
            self.learners_ensemble.fit_epochs(
                iterator=self.train_iterator,
                n_epochs=self.local_steps,
                weights=self.samples_weights,
            )
        for learner in self.learners_ensemble:
            if learner.lr_scheduler:
                learner.lr_scheduler.step()

    def step_record_update(self, single_batch_flag=True, *args, **kwargs):
        self.counter += 1
        # self.update_sample_weights()
        # self.update_learners_weights()
        if single_batch_flag:
            for _ in range(self.local_steps):
                batch = self.get_next_batch()
                client_updates = self.learners_ensemble.fit_batch_record_update(
                batch=batch, weights=self.samples_weights
                )

        else:
            client_updates = self.learners_ensemble[0].fit_epochs_record_update(
                iterator=self.train_iterator,
                n_epochs=self.local_steps,
                weights=self.samples_weights[0] * self.n_learners,
            )
        return client_updates

    def evaluate_iterator(self, iterator):
        if self.tune_locally:
            self.update_tuned_learners()

        learners_ensemble = (
            self.tuned_learners_ensemble
            if self.tune_locally
            else self.learners_ensemble
        )
        loss, acc = learners_ensemble.evaluate_iterator(iterator)
        return loss, acc

    def evaluate(self):
        train_loss, train_acc = self.evaluate_iterator(self.val_iterator)
        test_loss, test_acc = self.evaluate_iterator(self.test_iterator)
        self.logger.writerow([self.counter, f"{train_loss:.5f}", f"{train_acc:.5f}", f"{test_loss:.5f}", f"{test_acc:.5f}"])

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
        for learner_id, learner in enumerate(self.learners_ensemble):
            chkpts_path = os.path.join(
                dir_path, f"chkpts_client{client_id}_learner{learner_id}.pt"
            )
            learner.model.load_state_dict(torch.load(chkpts_path))

        learners_weights_path = os.path.join(
            dir_path, f"client{client_id}_learners_weights.npy"
        )
        self.learners_ensemble.learners_weights = np.load(
            learners_weights_path)





