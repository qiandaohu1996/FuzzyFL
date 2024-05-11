
class AGFLAggregator(Aggregator):
    r"""
    Implements
        `Adaptive Personalized Federated Learning`__(https://arxiv.org/abs/2003.13461)

    """

    def __init__(
        self,
        clients,
        global_learners_ensemble,
        log_freq,
        train_client_global_logger,
        test_client_global_logger,
        single_batch_flag,
        pre_rounds=50,
        comm_prob=0.2,
        sampling_rate=1.0,
        sample_with_replacement=False,
        test_clients=None,
        verbose=0,
        seed=None,
    ):
        super(AGFLAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            train_client_global_logger=train_client_global_logger,
            test_client_global_logger=test_client_global_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            single_batch_flag=single_batch_flag,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed,
        )
        # assert self.n_learners == 2, "APFL requires two learners"
        self.pre_rounds=pre_rounds
        self.n_clusters=self.n_learners-1
        self.write_logs()
        
    def pre_train(self):
        self.sample_clients()

        for client in self.sampled_clients:
            client.step(self.single_batch_flag)

        clients_learners = [client.learners_ensemble[0] for client in self.sampled_clients]
        average_learners(
            clients_learners,
            self.global_learners_ensemble[0],
            self.sampled_clients_weights,
        )
        print("global_learners_ensemble ",(self.global_learners_ensemble[0].model.parameters()[0].data[:5]))
        for learner in clients_learners:
            copy_model(learner.model, self.global_learners_ensemble[0].model)
            
    @calc_exec_time(calc_time=calc_time)
    def pre_clusting(self, n_clusters):
        print("\n============start clustring==============")
        
        clients_updates = np.zeros((self.n_sampled_clients, self.model_dim))
        for client_id, client in enumerate(self.sampled_clients):
            clients_updates[client_id] = client.step_record_update(self.single_batch_flag)
        similarities = pairwise_distances(clients_updates, metric="euclidean")
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="complete"
        )
        clustering.fit(similarities)
        self.clusters_indices = [
            np.argwhere(clustering.labels_ == i).flatten()
            for i in range(n_clusters)
        ]
        # print("clusters_indices ", self.clusters_indices)
        import sys

        print("=============cluster completed===============")
        print("\ndivided into {} clusters:".format(n_clusters))
        print('sys.stdout.width' )
        print(sys.stdout.width)
        for i, cluster in enumerate(self.clusters_indices):
            print(f"cluster {i}: {cluster}", end='', flush=True, width=sys.stdout.width)
        cluster_weights = torch.zeros(n_clusters)

        for cluster_id, indices in enumerate(self.clusters_indices):
            cluster_weights[cluster_id] = self.sampled_clients_weights[indices].sum()
            cluster_clients = [self.sampled_clients[i] for i in indices]

            average_learners(
                learners=[client.learners_ensemble[0]
                          for client in cluster_clients],
                target_learner=self.global_learners_ensemble[cluster_id+1],
                weights=self.sampled_clients_weights[indices]
                        / cluster_weights[cluster_id],
            )
            
        for client in self.clients:
            for learner_id, learner in enumerate(client.learners_ensemble[1:]):
                copy_model(learner.model, self.global_learners_ensemble[learner_id+1].model
                )

    def train(self):
        self.sample_clients()

        for clienr_id, client in enumerate(self.sampled_clients):
            # print("client_id", clienr_id)
            client.step(self.single_batch_flag)
            client_learners = client.learners_ensemble

            if client_learners.adaptive_alpha:
                client_learners.update_alpha()
                agfl_partial_average(client_learners[0], client_learners[1:], client_learners.alpha)

        if self.c_round % 5 == 0:
            for learner_id, learner in enumerate(self.global_learners_ensemble):
                learners = [client.learners_ensemble[learner_id] for client in self.clients]
                average_learners(learners, learner, weights=self.clients_weights) 
            
            for client in self.clients:
                average_learners(
                    learners=client.learners_ensemble[1:] ,
                    target_learner=client.learners_ensemble[0],
                    weights=self.clients_weights,
                )
        
        # for learner_id, learner in enumerate(self.global_learners_ensemble):
        #     learners = [client.learners_ensemble[learner_id] for client in self.clients]
        #     average_learners(learners, learner, weights=self.clients_weights)

        # assign the updated model to all clients
        # self.update_clients()

    def mix(self):
        if self.c_round < self.pre_rounds:
            self.pre_train()
            if self.c_round % self.log_freq == 0:
                self.write_logs()

        elif self.c_round == self.pre_rounds:
            self.pre_clusting(n_clusters=self.n_clusters)

        else:
            self.train()
            if self.c_round % self.log_freq == 0 or self.c_round == 199:
                self.write_logs()

        self.c_round += 1
 
        if self.c_round == 201:
            print("c_round==201")

    def update_clients(self):
        for client in self.clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(
                    learner.model, self.global_learners_ensemble[learner_id].model
                )

