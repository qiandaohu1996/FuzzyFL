import numpy as np


class FuzzyMScheduler:

    def __init__(self, initial_fuzzy_m):
        self.fuzzy_m = initial_fuzzy_m
        self.initial_fuzzy_m = initial_fuzzy_m

    def step(self, current_round=None):
        raise NotImplementedError

    def get_fuzzy_m(self):
        return self.fuzzy_m


class SqrtFuzzyMScheduler(FuzzyMScheduler):

    def step(self, current_round):
        self.fuzzy_m = 1 / np.sqrt(current_round) if current_round > 0 else 1


class LinearFuzzyMScheduler(FuzzyMScheduler):

    def step(self, current_round):
        self.fuzzy_m = 1 / current_round if current_round > 0 else 1
        return self.fuzzy_m


class ConstantFuzzyMScheduler(FuzzyMScheduler):

    def step(self, current_round):
        return self.fuzzy_m


class CosineAnnealingFuzzyMScheduler(FuzzyMScheduler):

    def __init__(self, initial_fuzzy_m, T_max, min_fuzzy_m, cycles=0):
        super().__init__(initial_fuzzy_m)
        self.T_max = T_max
        self.min_fuzzy_m = min_fuzzy_m
        self.cycles = cycles
        self.cycle = 1
        self.smooth_factor = 1

    def step(self, current_round):
        if current_round % self.T_max == 0:
            self.cycle += 1
            self.smooth_factor *= 0.9                    # you can adjust this value to control the smoothness

        current_round = current_round % (self.T_max)

        # multiply the cosine function by the current cycle
        self.fuzzy_m = self.initial_fuzzy_m + (0.5 * self.smooth_factor * (self.initial_fuzzy_m - self.min_fuzzy_m)) * \
            (np.cos(np.pi * (current_round / self.T_max)) - 1)

        if self.cycles > 0:
            if self.cycle > self.cycles:
                self.fuzzy_m = self.initial_fuzzy_m
        return self.fuzzy_m


class MultiStepFuzzyMScheduler(FuzzyMScheduler):

    def __init__(self, initial_fuzzy_m, milestones, gamma, min_fuzzy_m=1.5):
        super().__init__(initial_fuzzy_m)
        self.milestones = milestones
        self.gamma = gamma
        self.min_fuzzy_m = min_fuzzy_m

    def step(self, current_round):
        if current_round in self.milestones:
            self.fuzzy_m = (self.gamma - self.min_fuzzy_m) * self.gamma + self.min_fuzzy_m
        return self.fuzzy_m


class TwoMultiStepFuzzyMScheduler(FuzzyMScheduler):

    def __init__(self, initial_fuzzy_m, milestones1, milestones2, gamma, min_fuzzy_m=1.5):
        super().__init__(initial_fuzzy_m)
        self.milestones1 = milestones1
        self.milestones2 = milestones2
        self.gamma = gamma
        self.fuzzy_m = initial_fuzzy_m
        self.min_fuzzy_m = min_fuzzy_m

    def step(self, current_round):
        if current_round in self.milestones1:
            self.fuzzy_m = self.min_fuzzy_m + self.gamma
        elif current_round in self.milestones2:
            self.fuzzy_m = self.min_fuzzy_m - self.gamma
        return self.fuzzy_m


def get_fuzzy_m_scheduler(initial_fuzzy_m, scheduler_name="constant", n_rounds=200, min_fuzzy_m=1.5):

    min_fuzzy_m = initial_fuzzy_m - 0.2
    if scheduler_name == "sqrt":
        return SqrtFuzzyMScheduler(initial_fuzzy_m)

    elif scheduler_name == "linear":
        return LinearFuzzyMScheduler(initial_fuzzy_m)

    elif scheduler_name == "constant":
        return ConstantFuzzyMScheduler(initial_fuzzy_m)

    elif scheduler_name == "cosine_annealing":
        return CosineAnnealingFuzzyMScheduler(initial_fuzzy_m, T_max=n_rounds // 2, min_fuzzy_m=min_fuzzy_m)

    elif scheduler_name == "multi_step":
        assert n_rounds is not None, "Number of rounds is needed for \"multi_step\" scheduler!"
        min_fuzzy_m = initial_fuzzy_m - 0.2
        milestones = [n_rounds // 4, n_rounds // 2, 3 * (n_rounds // 4), 7 * (n_rounds // 8)]
        return MultiStepFuzzyMScheduler(initial_fuzzy_m, milestones=milestones, gamma=0.9, min_fuzzy_m=min_fuzzy_m)

    elif scheduler_name == "two_multi_step":
        assert n_rounds is not None, "Number of rounds is needed for \"two_multi_step\" scheduler!"
        milestones1 = [n_rounds // 4, 3 * (n_rounds // 8), n_rounds // 2]
        milestones2 = [5 * (n_rounds // 8), 3 * (n_rounds // 4), 7 * (n_rounds // 8)]
        return TwoMultiStepFuzzyMScheduler(
            initial_fuzzy_m, milestones1=milestones1, milestones2=milestones2, gamma=0.1, min_fuzzy_m=min_fuzzy_m
        )
