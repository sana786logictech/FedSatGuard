
def schedule(clients, trust):
    return sorted(clients, key=lambda c: trust.get(c.cid, 0.5), reverse=True)
