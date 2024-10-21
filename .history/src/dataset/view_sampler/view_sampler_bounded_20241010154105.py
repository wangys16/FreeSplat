from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor
import numpy as np
import os

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerBoundedCfg:
    name: Literal["bounded"]
    num_context_views: int
    num_target_views: int
    min_distance_between_context_views: int
    max_distance_between_context_views: int
    min_distance_to_context_views: int
    warm_up_steps: int
    initial_min_distance_between_context_views: int
    initial_max_distance_between_context_views: int
    random: bool = False
    extra: bool = False


class ViewSamplerBounded(ViewSampler[ViewSamplerBoundedCfg]):
    def schedule(self, initial: int, final: int) -> int:
        fraction = self.global_step / self.cfg.warm_up_steps
        return min(initial + int((final - initial) * fraction), final)

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        phase: int = 1,
        test_fvs: bool = False,
        path: str = None,
    ):
        num_views, _, _ = extrinsics.shape

        # Compute the context view spacing based on the current global step.
        if self.stage == "test":
           # When testing, always use the full gap.
           max_gap = self.cfg.max_distance_between_context_views
           min_gap = self.cfg.max_distance_between_context_views
        elif self.cfg.warm_up_steps > 0:
            max_gap = self.schedule(
                self.cfg.initial_max_distance_between_context_views,
                self.cfg.max_distance_between_context_views,
            )
            min_gap = self.schedule(
                self.cfg.initial_min_distance_between_context_views,
                self.cfg.min_distance_between_context_views,
            )
        else:
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.min_distance_between_context_views
        # print(f'max_gap: {max_gap}, min_gap: {min_gap}')
        # Pick the gap between the context views.
        if not self.cameras_are_circular:
            max_gap = min(num_views - 1, max_gap)
        min_gap = max(2 * self.cfg.min_distance_to_context_views, min_gap)
        if max_gap < min_gap:
            # print('not enough frame')
            raise ValueError("Example does not have enough frames!")
        context_gap = torch.randint(
            min_gap,
            max_gap + 1,
            size=tuple(),
            device=device,
        ).item()

        # print(f'context_gap: {context_gap}, min_gap: {min_gap}, max_gap: {max_gap}, num_views: {num_views}')

        if self.cfg.random:
            num_context_views = np.random.randint(2, self.num_context_views+1)
        else:
            num_context_views = self.num_context_views
        if (num_context_views > (num_views-1) // context_gap + 1) and not self.cfg.random:
            raise ValueError("Not enough views for the context views!")
        num_context_views = min(num_context_views, (num_views-1) // context_gap + 1)
        index_context_left = torch.randint(
                num_views if self.cameras_are_circular else num_views - context_gap*(num_context_views+phase-2),
                size=tuple(),
                device=device,
            ).item()
        index_start = index_context_left
        
        context_views_all = []
        index_target = []
        if num_context_views == 2:
            per_size = 4
        elif num_context_views == 3:
            per_size = 2
        else:
            per_size = 1
        for p in range(phase):
            context_views = [index_context_left]
            for i in range(num_context_views-1):
                index_context_right = context_views[i] + context_gap

                if self.is_overfitting:
                    index_context_left *= 0
                    index_context_right *= 0
                    index_context_right += max_gap

                # Pick the target view indices.
                index_target.append(torch.randint(
                        context_views[i] + self.cfg.min_distance_to_context_views,
                        index_context_right - self.cfg.min_distance_to_context_views,
                        size=(per_size,),
                        device=device,
                    ))
                context_views.append(index_context_right)
            
            
            index_context_left += context_gap
            context_views_all.append(context_views)

            if self.cameras_are_circular:
                index_target %= num_views
                index_context_right %= num_views
        
        return (
            torch.tensor(context_views_all, dtype=torch.int64, device=device),
            torch.cat(index_target),
            0
        )

    @property
    def num_context_views(self) -> int:
        # return 2
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
