"""PyTorch networks used in DQN policies."""
import torch


class FeedForwardLinearBlock(torch.nn.Module):
    """Simple feed-forward linear block layer with batchnorm and PReLU activation."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layer = torch.nn.Sequential(
                torch.nn.Linear(input_dim, output_dim),
                # torch.nn.BatchNorm1d(output_dim),
                torch.nn.LayerNorm(output_dim),
                torch.nn.PReLU(), # https://arxiv.org/pdf/1710.05941.pdf
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


# Example: https://gist.github.com/kkweon/52ea1e118101eb574b2a83b933851379
# Example: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class FeedForwardLinear(torch.nn.Module):
    """Deep Q-learning linear feed-forward network."""
    def __init__(self, dims: list[int]):
        super().__init__()

        # Preserve dimension list.
        self.dims = dims

        # Define list of layers.
        self.layers = torch.nn.ModuleList()

        # Build network using dimension list.
        # The input/output dimensions are collected by
        # zipping the original list with a shift-by-1 version.
        for input_dim, output_dim in zip(dims, dims[1:]):
            self.layers.append(FeedForwardLinearBlock(input_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Sequentially call forward functions of each intermediate layer.
        # This cascades the input through the entire network.
        for layer in self.layers:
            x = layer(x)

        # Return cascaded result.
        return x