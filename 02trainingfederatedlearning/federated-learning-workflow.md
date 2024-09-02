# Federated Learning Workflow and Code Implementation

## Overview

Federated Learning (FL) is a machine learning technique that trains an algorithm across multiple decentralized edge devices or servers holding local data samples, without exchanging them. Let's break down the workflow and see how it's implemented in the provided code.

## 1. Data Preparation and Distribution

In FL, data is naturally distributed across multiple clients. In this implementation, we simulate this distribution.

```python
trainset = datasets.MNIST("./MNIST_data/", download=True, train=True, transform=transform)
total_length = len(trainset)
split_size = total_length // 3
torch.manual_seed(42)
part1, part2, part3 = random_split(trainset, [split_size] * 3)

part1 = exclude_digits(part1, excluded_digits=[1, 3, 7])
part2 = exclude_digits(part2, excluded_digits=[2, 5, 8])
part3 = exclude_digits(part3, excluded_digits=[4, 6, 9])

train_sets = [part1, part2, part3]
```

This code splits the MNIST dataset into three parts, each excluding certain digits to create a non-IID distribution.

## 2. Model Initialization

The global model is initialized on the server.

```python
net = SimpleModel()
params = ndarrays_to_parameters(get_weights(net))
```

This creates an instance of the model and gets its initial parameters.

## 3. Client Setup

Clients are set up with their local data partitions.

```python
def client_fn(context: Context) -> Client:
    net = SimpleModel()
    partition_id = int(context.node_config["partition-id"])
    client_train = train_sets[int(partition_id)]
    client_test = testset
    return FlowerClient(net, client_train, client_test).to_client()

client = ClientApp(client_fn)
```

This function creates a client with a specific data partition based on the `partition-id`.

## 4. Server Configuration

The server is configured with the federated learning strategy.

```python
def server_fn(context: Context):
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        initial_parameters=params,
        evaluate_fn=evaluate,
    )
    config=ServerConfig(num_rounds=3)
    return ServerAppComponents(strategy=strategy, config=config)

server = ServerApp(server_fn=server_fn)
```

This sets up the server with the Federated Averaging (FedAvg) strategy and configures it to run for 3 rounds.

## 5. Training Rounds

The federated learning process occurs over multiple rounds. In each round:

### a. Client Selection

The server selects a subset of clients to participate in the current round. In this implementation, all clients participate in each round (`fraction_fit=1.0`).

### b. Model Distribution

The server sends the current global model to the selected clients.

### c. Local Training

Each selected client trains the model on its local data.

```python
def fit(self, parameters, config):
    set_weights(self.net, parameters)
    train_model(self.net, self.trainset)
    return get_weights(self.net), len(self.trainset), {}
```

This method in the `FlowerClient` class performs local training.

### d. Model Update Collection

Clients send their locally updated models back to the server.

### e. Global Aggregation

The server aggregates the updates from all clients to create a new global model. This is done using the FedAvg strategy.

## 6. Evaluation

After each round (or at specified intervals), the global model is evaluated.

```python
def evaluate(server_round, parameters, config):
    net = SimpleModel()
    set_weights(net, parameters)

    _, accuracy = evaluate_model(net, testset)
    _, accuracy137 = evaluate_model(net, testset_137)
    _, accuracy258 = evaluate_model(net, testset_258)
    _, accuracy469 = evaluate_model(net, testset_469)

    log(INFO, "test accuracy on all digits: %.4f", accuracy)
    log(INFO, "test accuracy on [1,3,7]: %.4f", accuracy137)
    log(INFO, "test accuracy on [2,5,8]: %.4f", accuracy258)
    log(INFO, "test accuracy on [4,6,9]: %.4f", accuracy469)

    if server_round == 3:
        cm = compute_confusion_matrix(net, testset)
        plot_confusion_matrix(cm, "Final Global Model")
```

This function evaluates the model on different subsets of the test data and logs the results.

## 7. Iteration

Steps 5-6 are repeated for the specified number of rounds (3 in this case).

## 8. Final Model

After all rounds are complete, the final global model is available on the server.

## 9. Simulation Execution

The entire federated learning process is executed as a simulation.

```python
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=3,
    backend_config=backend_setup,
)
```

This runs the simulation with the configured server and client applications.

## Key Points

1. **Privacy Preservation**: In this implementation, raw data never leaves the clients. Only model updates are shared.

2. **Non-IID Data Handling**: The data distribution simulates a real-world scenario where different clients have different data distributions.

3. **Scalability**: The system is designed to handle multiple clients, though in this simulation there are only three.

4. **Evaluation**: The server performs comprehensive evaluation on different subsets of the data, providing insights into the model's performance across different digit groups.

5. **Federated Averaging**: The FedAvg strategy is used for aggregating client updates, which is a standard approach in federated learning.

This implementation provides a clear example of how federated learning works, from data distribution to model training and evaluation, all while maintaining data privacy and handling non-IID data distributions.

