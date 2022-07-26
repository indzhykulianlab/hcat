from typing import Tuple, Dict, Optional, List
import torch
from torch import Tensor
from torch.autograd import Function
import torch.nn as nn
from typing import Tuple

import triton
import triton.language as tl


@triton.jit
def _embedding_forward_kernel(
        start_ptr,  # *Pointer* to first input vector
        output_ptr,  # *Pointer* to output vector

        # Vector Strides,
        c_stride, x_stride, y_stride, z_stride,

        # Probability Strides,
        n_prob_stride, x_prob_stride, y_prob_stride, z_prob_stride,

        # Centroid Number...
        N,

        # Centroids
        x_center, y_center, z_center,

        # Sigma
        x_sigma, y_sigma, z_sigma,

        # Size of the vector
        n_embed_elements,
        n_output_elements,

        # Constants
        BLOCK_SIZE: tl.constexpr
):
    """

    We want to move over the input tensor of shape [B, C=3, X, Y, Z]
    and do some gaussian operation then put it into an output tensor of shape [B, C=1, X, Y, Z]

    euclidean_norm = sqrt(Δx^2 + Δy^2 + Δz^2) where Δx = (x_embed - x_centroid_i)

                       /    (e_ix - C_kx)^2       (e_iy - C_ky)^2        (e_iz - C_kz)^2   \
    prob_k(e_i) = exp |-1 * ----------------  -  -----------------   -  ------------------  |
                      \     2*sigma_kx ^2         2*sigma_ky ^2          2 * sigma_kz ^2  /

    Returns:

    """
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to guard memory operations against out-of-bounds accesses
    embed_mask = offsets < n_embed_elements

    # Load in the vectors into memory
    x = tl.load(start_ptr + offsets, mask=embed_mask)  # Load values of input tensor in memory

    # Initialize intermediary tensors
    probability = tl.zeros(x.shape, dtype=tl.float32)
    sigma = tl.zeros(x.shape, dtype=tl.float32)
    centroid = tl.zeros(x.shape, dtype=tl.float32)

    # We first account for batching!
    c = offsets // c_stride  # Which channel are we in?
    _offsets = offsets - (c * c_stride)

    # Write the X Index
    x_ind = _offsets // x_stride
    _offsets = _offsets - (x_ind * x_stride)

    # Write the Y Index
    y_ind = _offsets // y_stride
    _offsets = _offsets - (y_ind * y_stride)

    # Write the Z Index
    z_ind = _offsets // z_stride

    output_offsets = (x_ind * x_prob_stride) + (y_ind * y_prob_stride) + (z_ind * z_prob_stride) + (N * n_prob_stride)
    output_masks = output_offsets < n_output_elements

    centroid = tl.where(c == 0, x_center, centroid)
    centroid = tl.where(c == 1, y_center, centroid)
    centroid = tl.where(c == 2, z_center, centroid)

    sigma = tl.where(c == 0, x_sigma, sigma)
    sigma = tl.where(c == 1, y_sigma, sigma)
    sigma = tl.where(c == 2, z_sigma, sigma)

    probability = ((x - centroid) * (x - centroid)) / (sigma + 1e-16)

    tl.atomic_add(output_ptr + output_offsets, probability, mask=output_masks)


@triton.jit
def _embedding_backward_kernel(

        previous_grad_ptr,  # [N ,X, Y, Z]  # Iterating over this vector because its the biggest!
        centroid_ptr,  # [N ,3] Use this to figure out which centroid?
        embed_ptr,  # *Pointer* to first input vector # [3, X, Y, Z]
        grad_ptr,  # *Pointer* to output vector # [3, X, Y, Z]

        # Vector Strides,
        n_stride, x_stride, y_stride, z_stride,

        # Centroid Strides
        n_centroid_stride, coord_stride,

        # Sigma
        sigma_ptr,

        # Size of the vector
        embed_numel,
        previous_grad_numel,
        centroid_numel,

        # Constants
        BLOCK_SIZE: tl.constexpr
):
    """
    Effectivly Does This...

    _embed_grad = torch.zeros(ctx.embed.shape, dtype=torch.float32, device=grad_outputs.device)
    sigma = torch.tensor(ctx.sigma, device=grad_outputs.device)
    for n, center in enumerate(ctx.centroids):
        _embed_grad += 2 * (ctx.embed - torch.tensor(center, device=grad_outputs.device).view(3, 1, 1, 1)) / sigma.view(3, 1,1,1) * grad_outputs[[n], ...]
    return _embed_grad, None, None


    """
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0
    coord_channel = tl.program_id(axis=1)  # We use a 1D launch grid so axis is 0

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    n_ind, x_ind, y_ind, z_ind = get_index(offsets, n_stride, x_stride, y_stride, z_stride)

    # Create a mask to guard memory operations against out-of-bounds accesses
    previous_grad_mask = offsets < previous_grad_numel
    previous_grad = tl.load(previous_grad_ptr + offsets,
                            mask=previous_grad_mask)  # Load values of input tensor in memory

    embed_offsets = (coord_channel * n_stride) + (x_ind * x_stride) + (y_ind * y_stride) + (z_ind * z_stride)
    embed_mask = embed_offsets < embed_numel
    embed = tl.load(embed_ptr + embed_offsets, mask=embed_mask)

    centroid_offsets = (n_ind * n_centroid_stride) + (coord_stride * coord_channel)
    centroid_mask = centroid_offsets < centroid_numel
    center = tl.load(centroid_ptr + centroid_offsets, mask=centroid_mask)

    sigma = tl.load(sigma_ptr + coord_channel)

    grad = 2 * (embed - center) / sigma * previous_grad
    tl.atomic_add(grad_ptr + embed_offsets, grad, mask=embed_mask)


@triton.jit
def get_index(offsets, c_stride, x_stride, y_stride, z_stride):
    # We first account for batching!
    c_ind = offsets // c_stride  # Which channel are we in?
    _offsets = offsets - (c_ind * c_stride)

    # Write the X Index
    x_ind = _offsets // x_stride
    _offsets = _offsets - (x_ind * x_stride)

    # Write the Y Index
    y_ind = _offsets // y_stride
    _offsets = _offsets - (y_ind * y_stride)

    # Write the Z Index
    z_ind = _offsets // z_stride

    return c_ind, x_ind, y_ind, z_ind


class embed2prob3D(Function):
    """
    Performs the vector to Embedding on 4D Inputs!
    """

    @staticmethod
    def forward(ctx, embed: torch.Tensor, sigma: Tuple[float], centroid: Tuple[Tuple[float]]):
        assert embed.ndim == 4
        assert embed.shape[0] == 3

        C, X, Y, Z = embed.shape
        _cs, _xs, _ys, _zs = embed.stride()

        N = len(centroid)

        output = torch.zeros((N, X, Y, Z), device=embed.device, dtype=embed.dtype)

        _cos, _xos, _yos, _zos = output.stride()

        # We need to preallocate the output

        assert embed.is_cuda and output.is_cuda
        assert embed.is_contiguous and output.is_contiguous

        n_embed_elements = embed.numel()
        n_output_elements = output.numel()

        for i, center in enumerate(centroid):
            grid = lambda META: (triton.cdiv(n_embed_elements, META['BLOCK_SIZE']),)

            _embedding_forward_kernel[grid](embed, output,

                                            # Input Strides
                                            c_stride=_cs, x_stride=_xs, y_stride=_ys, z_stride=_zs,

                                            # Output Strides
                                            n_prob_stride=_cos, x_prob_stride=_xos,
                                            y_prob_stride=_yos, z_prob_stride=_zos,

                                            # Which centroid are we?
                                            N=i,
                                            # Centers
                                            x_center=center[0], y_center=center[1], z_center=center[2],

                                            # Sigma
                                            x_sigma=sigma[0], y_sigma=sigma[1], z_sigma=sigma[2],

                                            # Number of Elements
                                            n_embed_elements=n_embed_elements,
                                            n_output_elements=n_output_elements,
                                            BLOCK_SIZE=513)

            torch.cuda.current_stream().synchronize()  # Neccessary to avoid any issues as the cuda stream may be very large!

        ctx.centroids = centroid
        ctx.embed = embed
        ctx.sigma = sigma

        return output

    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        """
        SUM_{n centroids} = 2 * (vec - x) * sigma * grad_outputs


        # Native pyTorch implementation...
        _embed_grad = torch.zeros(ctx.embed.shape, dtype=torch.float32, device=grad_outputs.device)
        sigma = torch.tensor(ctx.sigma, device=grad_outputs.device)
        for n, center in enumerate(ctx.centroids):
            _embed_grad += 2 * (ctx.embed - torch.tensor(center, device=grad_outputs.device).view(3, 1, 1, 1)) / sigma.view(3, 1,1,1) * grad_outputs[[n], ...]
        return _embed_grad, None, None

        :param ctx:
        :param grad_outputs:
        :return:
        """

        assert grad_outputs.ndim == 4
        assert ctx.embed.ndim == 4
        assert ctx.embed.shape[0] == 3
        assert len(ctx.centroids) == grad_outputs.shape[0]
        assert ctx.embed.shape[1] == grad_outputs.shape[1]
        assert ctx.embed.shape[2] == grad_outputs.shape[2]
        assert ctx.embed.shape[3] == grad_outputs.shape[3]
        assert len(ctx.sigma) == 3

        C, X, Y, Z = ctx.embed.shape
        _cs, _xs, _ys, _zs = ctx.embed.stride()

        output = torch.zeros((3, X, Y, Z), device=grad_outputs.device, dtype=grad_outputs.dtype)

        _cos, _xos, _yos, _zos = output.stride()

        previous_grad_numel = grad_outputs.numel()

        centroids = torch.tensor(ctx.centroids, device=grad_outputs.device)
        sigma = torch.tensor(ctx.sigma, device=grad_outputs.device)
        n_centroid_stride, coord_stride = centroids.stride()

        grid = lambda META: (triton.cdiv(previous_grad_numel, META['BLOCK_SIZE']), 3)  # 2D Lauch Grid!!!

        _embedding_backward_kernel[grid](
            previous_grad_ptr=grad_outputs,  # [N ,X, Y, Z]  # Iterating over this vector because its the biggest!
            centroid_ptr=centroids,  # [N ,3] Use this to figure out which centroid?
            embed_ptr=ctx.embed,  # *Pointer* to first input vector # [3, X, Y, Z]
            grad_ptr=output,  # *Pointer* to output vector # [3, X, Y, Z]

            # Vector Strides,
            n_stride=_cs, x_stride=_xs, y_stride=_ys, z_stride=_zs,

            # Centroid Strides
            n_centroid_stride=n_centroid_stride, coord_stride=coord_stride,

            # Sigma
            sigma_ptr=sigma,

            # Size of the vector
            embed_numel=ctx.embed.numel(),
            previous_grad_numel=grad_outputs.numel(),
            centroid_numel=centroids.numel(),

            # Constants
            BLOCK_SIZE=512)

        torch.cuda.current_stream().synchronize()

        return output, None, None


embed2prob = embed2prob3D.apply


class EmbeddingToProbability(nn.Module):
    def __init__(self):
        super(EmbeddingToProbability, self).__init__()

    def forward(self, embedding, centroids, sigma):

        if embedding.is_cuda():
            return self._forward_triton(embedding, centroids, sigma)
        else:
            return self._forward_torch(embedding, centroids, sigma)

    @torch.jit.ignore
    def _forward_triton(self, embedding, centroids, sigma):
        return torch.exp(embed2prob(embedding, centroids, sigma))

    def _forward_torch(self, embedding: Tensor, centroids: Tensor, sigma: Tensor) -> Tensor:
        """
        Calculates the euclidean distance between the centroid and the embedding
        embedding [B, 3, X, Y, Z] -> euclidean_norm[B, 1, X, Y, Z]
        euclidean_norm = sqrt(Δx^2 + Δy^2 + Δz^2) where Δx = (x_embed - x_centroid_i)

                             /    (e_ix - C_kx)^2       (e_iy - C_ky)^2        (e_iz - C_kz)^2   \
          prob_k(e_i) = exp |-1 * ----------------  -  -----------------   -  ------------------  |
                            \     2*sigma_kx ^2         2*sigma_ky ^2          2 * sigma_kz ^2  /


        Example:


        :param embedding: [B, K=3, X, Y, Z] embedding tensor where K is the likely centroid component: {X, Y, Z}
        :param centroids: [B, N, K_true=3] object centroids where N is the number of instances in the image and K_true is centroid {x, y, z}
        :param sigma: Tensor of shape = (1) or (embedding.shape)
        :return: [B, N, X, Y, Z] of probabilities for instance N
        """
        b, _, x, y, z = embedding.shape
        _, n, _ = centroids.shape

        # Centroids might be empty. In this case, return empty array!
        if n == 0:
            return torch.zeros((b, n, x, y, z), device=embedding.device)

        sigma = sigma + torch.tensor([1e-16], device=centroids.device)  # when sigma goes to zero, things tend to break

        if sigma.numel() == 1:
            sigma = torch.cat((sigma, sigma, sigma), dim=0)

        b, _, x, y, z = embedding.shape
        _, n, _ = centroids.shape
        # prob = torch.zeros((b, n, x, y, z), device=embedding.device)
        prob_list = torch.jit.annotate(List[Tensor], [])

        # Common operation. Done outside of loop for speed.
        newshape = (centroids.shape[0], 3, 1, 1, 1)
        sigma = sigma.pow(2).mul(2).view(newshape).mul(-1)

        # Calculate euclidean distance between centroid and embedding for each pixel and
        # turn that distance to probability and put it in preallocated matrix for each n
        # In eval mode uses in place operations to save memory!

        # A bit scuffed but should optimize well in torchscript
        for i in range(n):
            prob_list += [torch.exp(
                (embedding - centroids[:, i, :].view(newshape)).pow(2)
                    .div(sigma)
                    .sum(dim=1)
            ).squeeze(1)]
        prob = torch.stack(prob_list, dim=1)

        return prob

