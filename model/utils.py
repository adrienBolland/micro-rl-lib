from torch import nn


def create_mlp(input_size, output_size, layers, act_fun="ReLU", bias=True):

    act_fun = eval(f"nn.{act_fun}")

    layers_ = []
    for n_neurons in layers:
        # linear layers
        layers_.append(nn.Linear(input_size, n_neurons, bias))
        layers_.append(act_fun())

        input_size = n_neurons

    layers_.append(nn.Linear(input_size, output_size, bias))

    net_ = nn.Sequential(*layers_)

    return net_, layers_
