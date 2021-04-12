import torch


class LinearNet(torch.nn.Module):
    def __init__(self, input_size, output_size, intermediate_sizes: list = None):
        super(LinearNet, self).__init__()

        if intermediate_sizes is None or len(intermediate_sizes) == 0:
            layer_sizes = [input_size, output_size]
        else:
            layer_sizes = [input_size] + list(intermediate_sizes) + [output_size]

        layers = []
        for i in range(1, len(layer_sizes)):
            input_size_i, output_size_i = layer_sizes[i - 1], layer_sizes[i]
            bias = False if i == 1 else False  # no bias for the first layer, we have it in the features
            linear = torch.nn.Linear(input_size_i, output_size_i, bias=bias)
            layers.append(linear)
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        y_hat = self.layers(x)
        return y_hat

    def get_features(self, x) -> list:
        features = [x]
        for layer in self.layers:
            x = layer(x)
            features.append(x.detach().cpu())
        return features
