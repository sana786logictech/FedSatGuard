
class TrustModel:
    def __init__(self):
        self.trust = {}

    def update(self, cid, score):
        prev = self.trust.get(cid, 0.5)
        self.trust[cid] = 0.8 * prev + 0.2 * score
        return self.trust[cid]
