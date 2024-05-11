
from server.Aggregator import Aggregator
from utils.torch_utils import apfl_partial_average, average_learners, copy_model


class APFLAggregator(Aggregator):
    def __init__(
        self,
        clients,
        global_learners_ensemble,
        log_freq,
        train_client_global_logger,
        test_client_global_logger,
        single_batch_flag,
        sampling_rate=1,
        sample_with_replacement=False,
        test_clients=None,
        verbose=0,
        seed=None,
    ):
        super(APFLAggregator, self).__init__(
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
        assert self.n_learners == 2, "APFL requires two learners"

    def mix(self):
        self.sample_clients()
        print(self.single_batch_flag)
        for client in self.sampled_clients:
            client.step(self.single_batch_flag)
            learners = client.learners_ensemble

            if learners.adaptive_alpha:
                learners.update_alpha()
                apfl_partial_average(learners[0], learners[1], learners.alpha)

        if self.c_round % 5 == 0:
            average_learners(
                learners=[client.learners_ensemble[0] for client in self.clients],
                target_learner=self.global_learners_ensemble[0],
                weights=self.clients_weights,
            )
            # assign the updated model to all clients
            self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def write_logs(self):
        # 记录 本地训练精度 本地测试精度 平均值
        self._write_logs(self.train_client_global_logger, self.clients, "Train")

        if self.verbose > 0:
            print("-" * 50)

    def update_clients(self):
        for client in self.sampled_clients:
            copy_model(
                client.learners_ensemble[0].model,
                self.global_learners_ensemble[0].model,
            )
