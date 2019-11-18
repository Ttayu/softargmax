import torch
import torch.nn as nn
from torch.nn import functional as F


class SoftArgmax2D(nn.Module):
    """Creates a module that computes Soft-Argmax 2D of a given input heatmap.

    Returns the index of the maximum 2d coordinates of the give map.

    :param beta: The smoothing parameter.
    :param return_xy: The output order is [x, y].
    """

    def __init__(self, beta: int = 100, return_xy: bool = False):
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta: {beta}")
        super().__init__()
        self.beta = beta
        self.return_xy = return_xy

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        :param heatmap: The input heatmap is of size B x N x H x W.
        :return: The index of the maximum 2d coordinates is of size B x N x 2.
        """
        heatmap = heatmap.mul(self.beta)
        batch_size, num_channel, height, width = heatmap.size()
        device: str = heatmap.device

        softmax: torch.Tensor = F.softmax(
            heatmap.view(batch_size, num_channel, height * width), dim=2
        ).view(batch_size, num_channel, height, width)

        xx, yy = torch.meshgrid(list(map(torch.arange, [width, height])))

        approx_x = (
            softmax.mul(xx.float().to(device))
            .view(batch_size, num_channel, height * width)
            .sum(2)
            .unsqueeze(2)
        )
        approx_y = (
            softmax.mul(yy.float().to(device))
            .view(batch_size, num_channel, height * width)
            .sum(2)
            .unsqueeze(2)
        )

        output = [approx_x, approx_y] if self.return_xy else [approx_y, approx_x]
        output = torch.cat(output, 2)
        return output


class SoftArgmax3D(nn.Module):
    """Creates a module that computes Soft-Argmax 3D of a given input heatmap.

    Returns the index of the maximum 3d coordinates of the give map.

    :param beta: The smoothing parameter.
    :param return_xyz: The output order is [x, y, z].
    """

    def __init__(self, beta: int = 100, return_xyz: bool = False):
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta: {beta}")
        super().__init__()
        self.beta = beta
        self.return_xyz = return_xyz

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        :param heatmap: The input heatmap is of size B x N x D x H x W.
        :return: The index of the maximum 3d coordinates is of size B x N x 3.
        """
        batch_size, num_channel, depth, height, width = heatmap.size()
        volume_size = depth * height * width
        device = heatmap.device

        heatmap = heatmap.mul(self.beta)
        softmax = F.softmax(
            heatmap.view(batch_size, num_channel, volume_size), dim=2
        ).view(batch_size, num_channel, depth, height, width)

        zz, yy, xx = torch.meshgrid(list(map(torch.arange, [depth, height, width])))

        approx_x = (
            softmax.mul(xx.float().to(device))
            .view(batch_size, num_channel, volume_size)
            .sum(2)
            .unsqueeze(2)
        )
        approx_y = (
            softmax.mul(yy.float().to(device))
            .view(batch_size, num_channel, volume_size)
            .sum(2)
            .unsqueeze(2)
        )
        approx_z = (
            softmax.mul(zz.float().to(device))
            .view(batch_size, num_channel, volume_size)
            .sum(2)
            .unsqueeze(2)
        )

        output = (
            [approx_x, approx_y, approx_z]
            if self.return_xyz
            else [approx_z, approx_y, approx_x]
        )
        output = torch.cat(output, 2)
        return output
