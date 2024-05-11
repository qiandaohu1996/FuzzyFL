
from server.PersonalizedAgg import PersonalizedAggregator
from utils.torch_utils import average_learners, partial_average


class LoopLessLocalSGDAggregator(PersonalizedAggregator):
    """
    Implements L2SGD introduced in
    'Federated Learning of a Mixture of Global and Local Models'__. (https://arxiv.org/pdf/2002.05516.pdf)


    """

    def __init__(
        self,
        clients,
        global_learners_ensemble,
        log_freq,
        train_client_global_logger,
        test_client_global_logger,
        single_batch_flag,
        comm_prob=0.2,
        penalty_parameter=0.1,
        sampling_rate=1.0,
        sample_with_replacement=False,
        test_clients=None,
        verbose=0,
        seed=None,
    ):
        super(LoopLessLocalSGDAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
            single_batch_flag=single_batch_flag,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
        )

        self.comm_prob = comm_prob
        self.penalty_parameter = penalty_parameter

    @property
    def comm_prob(self):
        return self.__comm_prob

    @comm_prob.setter
    def comm_prob(self, comm_prob):
        self.__comm_prob = comm_prob

    def mix(self):
        comm_flag = self.np_rng.binomial(1, self.comm_prob, 1)

        if comm_flag:
            for learner_id, learner in enumerate(self.global_learners_ensemble):
                learners = [
                    client.learners_ensemble[learner_id] for client in self.clients
                ]
                average_learners(learners, learner, weights=self.clients_weights)

                partial_average(
                    learners=learners,
                    average_learner=learner,
                    alpha=self.penalty_parameter / self.comm_prob,
                )

        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

        else:
            for client in self.clients:
                client.step(single_batch_flag=True)