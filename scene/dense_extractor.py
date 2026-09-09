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
import torch.nn.functional as F
import copy

from model_store import get_model_store
from scene.extractor_model import XFeatModel

class DenseExtractor():
    """Extract dense feature maps from an image."""
    @torch.no_grad()
    def __init__(self, width, height):
        model_store = get_model_store()
        cache_name = f"dense_extractor_{width}_{height}.pt"
        cache_path = model_store.jit_path(cache_name)
        dummy_img = torch.randn(1, 3, height, width).cuda().to(torch.half)
        
        if cache_path.is_file():
            self.extractor = torch.jit.load(str(cache_path))
        else:
            self.extractor = model_store.load_xfeat(top_k=4096)
            self.extractor = self.extractor.half().cuda().eval()

            state_dict = copy.deepcopy(self.extractor.state_dict())
            self.extractor.net = XFeatModel(4).half().cuda().eval()
            self.extractor.load_state_dict(state_dict)
            def preprocess_tensor(x):
                H, W = x.shape[-2:]
                _H, _W = (H//32) * 32, (W//32) * 32
                rh, rw = H/_H, W/_W

                x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=True)
                return x, rh, rw
            
            @torch.no_grad()
            def extract(x):
                _, _, H, W = x.shape
                x, rh1, rw1 = preprocess_tensor(x)
                
                M1 = self.extractor.net(x)

                M1 = F.normalize(M1, dim=1)
                M1 = M1[0].permute(1, 2, 0)
                return M1

            self.extractor.forward = extract

            self.extractor = torch.jit.trace(self.extractor, [dummy_img])
            self.extractor = torch.jit.script(self.extractor)
            model_store.save_jit(self.extractor, cache_name)

        self.extractor(torch.rand_like(dummy_img))

    @torch.no_grad()
    def __call__(self, image):
        return self.extractor(image[None].half())
