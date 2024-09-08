import torch
import torch.nn as nn

from . import functional as F

__all__ = ['BallQuery']


class BallQuery(nn.Module):
    def __init__(self, radius, num_neighbors, include_coordinates=True):
        super().__init__()
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.include_coordinates = include_coordinates

<<<<<<< Updated upstream
    def forward(self, points_coords, centers_coords, temb, points_features=None):
=======
    def forward(self, points_coords, centers_coords, points_features=None):
>>>>>>> Stashed changes
        points_coords = points_coords.contiguous()
        centers_coords = centers_coords.contiguous()
        neighbor_indices = F.ball_query(centers_coords, points_coords, self.radius, self.num_neighbors)
        neighbor_coordinates = F.grouping(points_coords, neighbor_indices)
        neighbor_coordinates = neighbor_coordinates - centers_coords.unsqueeze(-1)

        if points_features is None:
            assert self.include_coordinates, 'No Features For Grouping'
            neighbor_features = neighbor_coordinates
        else:
            neighbor_features = F.grouping(points_features, neighbor_indices)
            if self.include_coordinates:
                neighbor_features = torch.cat([neighbor_coordinates, neighbor_features], dim=1)
<<<<<<< Updated upstream
        return neighbor_features, F.grouping(temb, neighbor_indices)
=======
        return neighbor_features
>>>>>>> Stashed changes

    def extra_repr(self):
        return 'radius={}, num_neighbors={}{}'.format(
            self.radius, self.num_neighbors, ', include coordinates' if self.include_coordinates else '')
