
import numpy as np

class Client:
    def __init__(self, cid, malicious=False):
        self.cid = cid
        self.malicious = malicious

    def local_train(self, model):
        grad = np.random.randn(len(model)) * 0.1

        if self.malicious:
            grad = grad * -3  # poisoning

        trust = np.exp(-np.linalg.norm(grad))
        return grad, trust
