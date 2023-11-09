from itertools import permutations

import torch


EPS = 1e-8


def cal_loss(source, estimate_source, source_lengths):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    max_sdr, perms, max_sdr_idx, sdr_set = cal_si_sdr_with_pit(source,
                                                               estimate_source,
                                                               source_lengths)
    loss = 0 - torch.mean(max_sdr)

    reorder_estimate_source = reorder_source(
        estimate_source, perms, max_sdr_idx)
    return loss, max_sdr, estimate_source, reorder_estimate_source


def cal_si_sdr_with_pit(source, estimate_source, source_lengths):
    source = source.squeeze(dim=2)
    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2,
                              keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target,
                              dim=3, keepdim=True)  # [B, C, C, 1]
    s_target_energy = torch.sum(
        s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    pair_wise_si_sdr = torch.sum(
        pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_sdr = 10 * torch.log10(pair_wise_si_sdr + EPS)  # [B, C, C]
    pair_wise_si_sdr = torch.transpose(pair_wise_si_sdr, 1, 2)

    # permutations, [C!, C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    sdr_set = torch.einsum('bij,pij->bp', [pair_wise_si_sdr, perms_one_hot])
    max_sdr_idx = torch.argmax(sdr_set, dim=1)  # [B]
    max_sdr, _ = torch.max(sdr_set, dim=1, keepdim=True)
    max_sdr /= C
    return max_sdr, perms, max_sdr_idx, sdr_set / C


def reorder_source(source, perms, max_sdr_idx):
    B, C, *_ = source.size()

    max_sdr_perm = torch.index_select(perms, dim=0, index=max_sdr_idx)
    reorder_source = torch.zeros_like(source)
    for b in range(B):
        for c in range(C):
            reorder_source[b, c] = source[b, max_sdr_perm[b][c]]
    return reorder_source


def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, _, T = source.size()
    mask = source.new_ones((B, 1, T))
    for i in range(B):
        mask[i, :, source_lengths[i]:] = 0
    return mask