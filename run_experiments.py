
import numpy as np
from fl_simulation.server import Server
from fl_simulation.client import Client

clients = [
    Client(i, malicious=(i % 4 == 0)) for i in range(10)
]

server = Server(clients)
model = server.train(5)

print("Final Model:", model)
