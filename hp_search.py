import nevergrad as ng

def fake_training(learning_rate: float, batch_size: int, architecture: str) -> float:
    # optimal for learning_rate=0.2, batch_size=4, architecture="conv"
    return (learning_rate - 0.2)**2 + (batch_size - 4)**2 + (0 if architecture == "conv" else 10)

# Instrumentation class is used for functions with multiple inputs
# (positional and/or keywords)
parametrization = ng.p.Instrumentation(
    # a log-distributed scalar between 0.001 and 1.0
    learning_rate=ng.p.Log(lower=1e-3, upper=1e-4),
    # an integer from 1 to 12
    batch_size=ng.p.Scalar(lower=1, upper=12).set_integer_casting(),
    # either "conv" or "fc"
    optimizer=ng.p.Choice(["conv", "fc"])
)

optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=100)
recommendation = optimizer.minimize(fake_training)

# show the recommended keyword arguments of the function
print(recommendation.kwargs)
>>> {'learning_rate': 0.1998, 'batch_size': 4, 'architecture': 'conv'}