from server import Aggregator


class NoCommunicationAggregator(Aggregator):
    def mix(self):
        self.sample_clients()
        for client in self.sampled_clients:
            client.step(self.single_batch_flag)
        self.c_round += 1
        if self.c_round % self.log_freq == 0:
            self.evaluate()

    def update_clients(self):
        pass
