import torch
from einops import repeat
from jaxtyping import Int
from torch import Tensor

Index = Int[Tensor, "n n-1"]


def generate_heterogeneous_index(
    n: int,
    device: torch.device = torch.device("cpu"),
) -> tuple[Index, Index]:
    """Generate indices for all pairs except self-pairs."""
    arange = torch.arange(n, device=device)

    
    # Generate an index that represents the item itself.
    index_self = repeat(arange, "h -> h w", w=n - 1)

    # Generate an index that represents the other items.
    index_other = repeat(arange, "w -> h w", h=n).clone()
    index_other += torch.ones((n, n), device=device, dtype=torch.int64).triu()
    if n <= 3:
        index_other = index_other[:, :-1]
        return index_self, index_other
    else:
        n_views = n
        num_context_views = 2
        cur_indices = torch.arange(n_views, device=device)
        full_indices = torch.arange(n_views, device=device)[None].repeat(n_views,1)
                        
        slide_mask = torch.zeros((n_views, n_views), dtype=torch.bool, device=full_indices.device)
        for i in range(n_views):
            if i < num_context_views // 2:
                start = 0
                end = min(num_context_views, n_views)
            elif i >= n_views - num_context_views // 2-1:
                start = max(n_views - num_context_views - 1, 0)
                end = n_views - 1 
            else:
                start = max(i - (num_context_views) // 2, 0)
                end = min(i + (num_context_views-1) // 2 + 1, n_views)
            try:
                assert end - start == min(num_context_views, n_views)
            except:
                print('error of slide mask:', slide_mask)
                exit(1)
            slide_mask[i, start:end] = 1
        # print('slide_mask:', slide_mask)
        src_indices = full_indices[slide_mask].view(n_views,min(n_views, num_context_views))
        index_self = torch.gather(index_self, 1, src_indices)
        index_other = torch.gather(index_other, 1, src_indices)
    
    print('normal index_self:', index_self)
    print('normal index_other:', index_other)

    return index_self, index_other


def generate_heterogeneous_index_transpose(
    n: int,
    device: torch.device = torch.device("cpu"),
) -> tuple[Index, Index]:
    """Generate an index that can be used to "transpose" the heterogeneous index.
    Applying the index a second time inverts the "transpose."
    """
    if n <= 3:
        arange = torch.arange(n, device=device)
        ones = torch.ones((n, n), device=device, dtype=torch.int64)

        # if n <= 3:
        index_self = repeat(arange, "w -> h w", h=n).clone()
        index_self = index_self + ones.triu()

        index_other = repeat(arange, "h -> h w", w=n)
        index_other = index_other - (1 - ones.triu())

        return index_self[:, :-1], index_other[:, :-1]
    elif n == 10:
        index_self = torch.tensor([[1, 1],
                [0, 2],
                [1, 3],
                [2, 4],
                [3, 5],
                [4, 6],
                [5, 7],
                [6, 8],
                [7, 9],
                [7, 7]])
        index_other = torch.tensor([[0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 1]])
        return index_self, index_other
    else:
        arange = torch.arange(3, device=device)
        ones = torch.ones((3, 3), device=device, dtype=torch.int64)

        # if n <= 3:
        index_self = repeat(torch.arange(n, device=device), "w -> h w", h=n).clone()
        index_self = index_self + torch.ones((n, n), device=device, dtype=torch.int64).triu()

        index_other = repeat(arange, "h -> h w", w=3)
        index_other = index_other - (1 - ones.triu())

    
    
    n_views = n
    num_context_views = 2
    cur_indices = torch.arange(n_views, device=device)
    full_indices = torch.arange(n_views, device=device)[None].repeat(n_views,1)
                    
    slide_mask = torch.zeros((n_views, n_views), dtype=torch.bool, device=full_indices.device)
    for i in range(n_views):
        if i < num_context_views // 2:
            start = 0
            end = min(num_context_views, n_views)
        elif i >= n_views - num_context_views // 2-1:
            start = max(n_views - num_context_views - 1, 0)
            end = n_views - 1 
        else:
            start = max(i - (num_context_views) // 2, 0)
            end = min(i + (num_context_views-1) // 2 + 1, n_views)
        try:
            assert end - start == min(num_context_views, n_views)
        except:
            print('error of slide mask:', slide_mask)
            exit(1)
        slide_mask[i, start:end] = 1
    # print('slide_mask:', slide_mask)
    src_indices = full_indices[slide_mask].view(n_views,min(n_views, num_context_views))
    index_self = torch.gather(index_self, 1, src_indices)

    n_views = 3
    num_context_views = 2
    cur_indices = torch.arange(n_views, device=device)
    full_indices = torch.arange(n_views, device=device)[None].repeat(n_views,1)
                    
    slide_mask = torch.zeros((n_views, n_views), dtype=torch.bool, device=full_indices.device)
    for i in range(n_views):
        if i < num_context_views // 2:
            start = 0
            end = min(num_context_views, n_views)
        elif i >= n_views - num_context_views // 2-1:
            start = max(n_views - num_context_views - 1, 0)
            end = n_views - 1 
        else:
            start = max(i - (num_context_views) // 2, 0)
            end = min(i + (num_context_views-1) // 2 + 1, n_views)
        try:
            assert end - start == min(num_context_views, n_views)
        except:
            print('error of slide mask:', slide_mask)
            exit(1)
        slide_mask[i, start:end] = 1
    # print('slide_mask:', slide_mask)
    src_indices = full_indices[slide_mask].view(n_views,min(n_views, num_context_views))
    index_other = torch.gather(index_other, 1, src_indices)

    # index_self = torch.cat([index_self[:1]]+[index_self[1:2] for _ in range(n-2)]+[index_self[-1:]], dim=0)
    index_other = torch.cat([index_other[:1]]+[index_other[1:2] for _ in range(n-2)]+[index_other[-1:]], dim=0)

    # print('transpose index_self:', index_self)
    # print('transpose index_other:', index_other)

    return index_self, index_other
