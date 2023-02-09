import torch


class ResLayer(torch.nn.Module):
    def __init__(self, in_units, width):

        super(ResLayer, self).__init__()
        self.linear = torch.nn.Linear(width, width)
        self.layer_norm = torch.nn.LayerNorm(in_units)

    def forward(self, x):

        out = self.layer_norm(x)
        out = torch.nn.functional.relu(out)
        out = self.linear(out)

        return x + out


def make_network(input_dim=5, width=64, L=2, out_features=5):

    network = torch.nn.Sequential(
        torch.nn.Linear(input_dim, width),
        *[ResLayer(width, width) for _ in range(L)],
        torch.nn.LayerNorm(width),
        torch.nn.Linear(width, out_features)
    )
    return network
