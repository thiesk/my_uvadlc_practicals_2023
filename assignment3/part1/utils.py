################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import torch
from torchvision.utils import make_grid
import numpy as np


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    assert not (std < 0).any().item(), "The reparameterization trick got a negative std as input. " + \
                                       "Are you sure your input is std and not log_std?"
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # sample noise
    epsilon = torch.randn_like(std)

    # re-parametrisation trick
    z = mean + (std * epsilon)

    #######################
    # END OF YOUR CODE    #
    #######################
    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See the definition of the regularization loss in Section 1.4 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # implementing formula from the pdf

    # part inside the sum
    summand = 0.5 * ((2 * log_std).exp() + mean ** 2 - 1 - 2 * log_std)

    # sum and dividing by 2
    KLD = torch.sum(input=summand, dim=-1)

    #######################
    # END OF YOUR CODE    #
    #######################
    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    B, C, H, W = img_shape
    # use the formula on pdf, exlcuding batch dimension when using img_shape
    # math: log_2(e) = 1/ ln(2)
    bpd = elbo / (np.log(2) * (C * H * W))
    #######################
    # END OF YOUR CODE    #
    #######################
    return bpd



@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the distance
                    between different latents in percentiles is 1/grid_size
    Outputs:
        img_grid - Grid of images representing the manifold.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    ## Hints:
    # - You can use the icdf method of the torch normal distribution  to obtain z values at percentiles.
    # - Use the range [0.5/grid_size, 1.5/grid_size, ..., (grid_size-0.5)/grid_size] for the percentiles.
    percentiles = [(i-0.5) / grid_size for i in range(1, grid_size + 1)]
    # - torch.meshgrid might be helpful for creating the grid of values
    x_latent, y_latent = torch.meshgrid(percentiles, percentiles, indexing="xy")
    normal_dist = torch.distributions.Normal(0, 1)
    z_1 = normal_dist.icdf(x_latent)
    z_2 = normal_dist.icdf(y_latent)

    # put together for make grid
    z = torch.stack([z_1, z_2], dim=-1)
    z = torch.flatten(z, end_dim=-2)

    logit_z = decoder(z) # see what i did there

    # - Remember to apply a softmax after the decoder
    # get probs from logits
    prob_z = torch.nn.functional.softmax(logit_z, dim=1)
    # sort, st. they work with the categorical sampling
    prob_z = torch.permute(prob_z, (0, 2, 3, 1))
    prob_z = torch.flatten(prob_z, end_dim=2)

    # sample categorically
    imgs = torch.multinomial(prob_z, 1)
    # put back to image batch format
    imgs = imgs.reshape(-1, 28, 28, 1)
    imgs = torch.permute(imgs, (0, 3, 1, 2))

    # - You can use torchvision's function "make_grid" to combine the grid_size**2 images into a grid
    img_grid = make_grid(imgs, nrow=grid_size).float()

    #######################
    # END OF YOUR CODE    #
    #######################

    return img_grid

