
from copy import deepcopy
from client.Client import Client
from utils.constants import LOCAL_HEAD_UPDATES
from utils.torch_utils import copy_model


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
