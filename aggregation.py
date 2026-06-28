
import numpy as np

def trust_weighted_aggregation(updates, trusts):
    trusts = np.array(trusts)
    updates = np.array(updates)

    weights = trusts / (np.sum(trusts) + 1e-9)
    return np.sum(updates * weights[:, None], axis=0)
