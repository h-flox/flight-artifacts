from typing import List, Tuple
from time import time
import flwr as fl
from flwr.common import Metrics
import argparse

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}



# Get partition id
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--clients",
    required=True,
    type=int,
    help="Number of clientsi")

#num_clients = parser.parse_args().clients
parser.add_argument(
    "--ip",
    required=True,
    type=str,
    help="ip address")

parser.add_argument(
    "--model",
    default=18,
    type=int,
    help="model num/id"
)

args = parser.parse_args()
# Define strategy
strategy = fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average, 
        min_available_clients=args.clients, 
        min_fit_clients=args.clients)



# Start Flower server
fl.server.start_server(
    server_address=args.ip+":9898",
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=strategy,
)



end = time()*1_000_000_000 #this to get it into nanoseconds
with open(f'flower_time_{args.clients}_{args.model}.txt','w') as f:
    f.write("{:0.8f}".format(end))
