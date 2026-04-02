#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import cupy
import math
from enum import Enum
from poses.mini_ba import MiniBA
from utils import pts2px, sixD2mtx


class EstimatorType(Enum):
    FUNDAMENTAL_8PTS = 0
    P4P = 1


class RANSACEstimator:
    @torch.no_grad()
    def __init__(self, N: int, max_error: float, type: EstimatorType):
        """
        Initialize the RANSAC estimator.

        Args:
            N (int): Number of models to estimate.
            max_error (float): Maximum reprojection error for inliers.
            type (EstimatorType): Type of estimator to use.
        """
        self.N = N
        self.max_error = max_error
        self.type = type

        # Read the CUDA source code and set the include directory to poses/
        with open("poses/ransac.cu", "r") as f:
            cuda_source = f.read()
        self.module = cupy.RawModule(
            code=cuda_source,
            options=("--std=c++17", "-Iposes"),
        )
