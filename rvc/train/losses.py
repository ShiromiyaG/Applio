import torch
import torch.nn.functional as F

from rvc.lib.algorithm.san import SAN_DIRECTION_WEIGHT


def _branch_weight(branch_weights, index):
    """``branch_weights[index]`` as a float, or 1.0 when none are given.

    A plain Python number rather than a tensor: these are per-branch constants
    read once off the discriminator, and keeping them out of the graph is what
    makes an unweighted call identical to the code that had no weighting.
    """
    if branch_weights is None:
        return 1.0
    return float(branch_weights[index])


def feature_loss(fmap_r, fmap_g, branch_weights=None):
    """
    Compute the feature loss between reference and generated feature maps.

    A branch's weight scales every layer of it, because its feature-matching
    pull and its adversarial pull are the same head's opinion -- discounting
    only one would leave the generator chasing features from a head it was
    told not to believe.

    Args:
        fmap_r (list of torch.Tensor): List of reference feature maps.
        fmap_g (list of torch.Tensor): List of generated feature maps.
        branch_weights (sequence of float, optional): One weight per branch, in
            the discriminator's own order. Defaults to None (all 1.0).
    """
    return 2 * sum(
        _branch_weight(branch_weights, i) * torch.mean(torch.abs(rl - gl))
        for i, (dr, dg) in enumerate(zip(fmap_r, fmap_g))
        for rl, gl in zip(dr, dg)
    )


def discriminator_loss(disc_real_outputs, disc_generated_outputs, branch_weights=None):
    """
    Compute the discriminator loss for real and generated outputs.

    A ``v4`` branch run with ``san_training=True`` hands in a
    ``(function, direction)`` pair instead of one logit -- see
    ``rvc.lib.algorithm.san``.  Both halves take the same one-sided, bounded
    surrogate: mirroring the function term keeps the fake-direction term
    saturating, where the unbounded form lets the discriminator win by pushing
    the direction output negative without discriminating at all.

    Weighting the discriminator's own loss as well as the generator's is
    deliberate: a head the generator is told to discount but that still trains
    at full rate keeps pulling away, and that gap is what the weight closes.

    Args:
        disc_real_outputs (list of torch.Tensor): List of discriminator outputs for real samples.
        disc_generated_outputs (list of torch.Tensor): List of discriminator outputs for generated samples.
        branch_weights (sequence of float, optional): One weight per branch, in
            the discriminator's own order. Defaults to None (all 1.0).
    """
    loss = 0
    r_losses = []
    g_losses = []
    for i, (dr, dg) in enumerate(zip(disc_real_outputs, disc_generated_outputs)):
        if isinstance(dr, (list, tuple)):
            dr_fun, dr_dir = dr
            dg_fun, dg_dir = dg
            r_loss = torch.mean(
                F.softplus(1 - dr_fun.float()) ** 2
            ) + SAN_DIRECTION_WEIGHT * torch.mean(F.softplus(1 - dr_dir.float()) ** 2)
            g_loss = torch.mean(
                F.softplus(dg_fun.float()) ** 2
            ) + SAN_DIRECTION_WEIGHT * torch.mean(F.softplus(dg_dir.float()) ** 2)
        else:
            r_loss = torch.mean((1 - dr.float()) ** 2)
            g_loss = torch.mean(dg.float() ** 2)

        # r_losses.append(r_loss.item())
        # g_losses.append(g_loss.item())
        weight = _branch_weight(branch_weights, i)
        loss += weight * (r_loss + g_loss) if weight != 1.0 else r_loss + g_loss

    return loss, r_losses, g_losses


def generator_loss(disc_outputs, use_softplus: bool = False, branch_weights=None):
    """
    Compute the generator loss based on discriminator outputs.

    ``use_softplus`` is SAN's generator objective.  The outputs here are always
    plain logits -- the generator pass never asks for the direction, which is
    the discriminator's business alone -- so this changes the surrogate, not the
    shape.

    Args:
        disc_outputs (list of torch.Tensor): List of discriminator outputs for generated samples.
        use_softplus (bool): Use the squared-softplus surrogate SAN pairs with.
        branch_weights (sequence of float, optional): One weight per branch, in
            the discriminator's own order. Defaults to None (all 1.0).
    """
    loss = 0
    gen_losses = []
    for i, dg in enumerate(disc_outputs):
        if use_softplus:
            l = torch.mean(F.softplus(1.0 - dg.float()).square())
        else:
            l = torch.mean((1 - dg.float()) ** 2)
        # gen_losses.append(l.item())
        weight = _branch_weight(branch_weights, i)
        loss += weight * l if weight != 1.0 else l

    return loss, gen_losses


def discriminator_loss_scaled(disc_real, disc_fake, scale=1.0):
    loss = 0
    for i, (d_real, d_fake) in enumerate(zip(disc_real, disc_fake)):
        real_loss = torch.mean((1 - d_real) ** 2)
        fake_loss = torch.mean(d_fake**2)
        _loss = real_loss + fake_loss
        loss += _loss if i < len(disc_real) / 2 else scale * _loss
    return loss, None, None


def generator_loss_scaled(disc_outputs, scale=1.0):
    loss = 0
    for i, d_fake in enumerate(disc_outputs):
        d_fake = d_fake.float()
        _loss = torch.mean((1 - d_fake) ** 2)
        loss += _loss if i < len(disc_outputs) / 2 else scale * _loss
    return loss, None, None


def discriminator_loss_scaled(disc_real, disc_fake, scale=1.0):
    """
    Compute the scaled discriminator loss for real and generated outputs.

    Args:
        disc_real (list of torch.Tensor): List of discriminator outputs for real samples.
        disc_fake (list of torch.Tensor): List of discriminator outputs for generated samples.
        scale (float, optional): Scaling factor applied to losses beyond the midpoint. Default is 1.0.
    """
    midpoint = len(disc_real) // 2
    losses = []
    for i, (d_real, d_fake) in enumerate(zip(disc_real, disc_fake)):
        real_loss = (1 - d_real).pow(2).mean()
        fake_loss = d_fake.pow(2).mean()
        total_loss = real_loss + fake_loss
        if i >= midpoint:
            total_loss *= scale
        losses.append(total_loss)
    loss = sum(losses)
    return loss, None, None


def generator_loss_scaled(disc_outputs, scale=1.0):
    """
    Compute the scaled generator loss based on discriminator outputs.

    Args:
        disc_outputs (list of torch.Tensor): List of discriminator outputs for generated samples.
        scale (float, optional): Scaling factor applied to losses beyond the midpoint. Default is 1.0.
    """
    midpoint = len(disc_outputs) // 2
    losses = []
    for i, d_fake in enumerate(disc_outputs):
        loss_value = (1 - d_fake).pow(2).mean()
        if i >= midpoint:
            loss_value *= scale
        losses.append(loss_value)
    loss = sum(losses)
    return loss, None, None


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    Compute the Kullback-Leibler divergence loss.

    Args:
        z_p (torch.Tensor): Latent variable z_p [b, h, t_t].
        logs_q (torch.Tensor): Log variance of q [b, h, t_t].
        m_p (torch.Tensor): Mean of p [b, h, t_t].
        logs_p (torch.Tensor): Log variance of p [b, h, t_t].
        z_mask (torch.Tensor): Mask for the latent variables [b, h, t_t].
    """
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2 * logs_p)
    kl = (kl * z_mask).sum()
    loss = kl / z_mask.sum()
    return loss
