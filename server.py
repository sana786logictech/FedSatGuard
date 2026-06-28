
import numpy as np
from fl_simulation.aggregation import trust_weighted_aggregation

class Server:
    def __init__(self, clients):
        self.clients = clients
        self.global_model = np.zeros(10)

    def train(self, rounds=5):
        for r in range(rounds):
            updates = []
            trusts = []

            for c in self.clients:
                u, t = c.local_train(self.global_model)
                updates.append(u)
                trusts.append(t)

            self.global_model = trust_weighted_aggregation(updates, trusts)

        return self.global_model
