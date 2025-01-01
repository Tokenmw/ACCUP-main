import numpy as np
import torch
from utils.utils import fix_randomness

def DataTransform(sample, config, seed_id):
    fix_randomness(seed_id)
    cur_dataseet_name = config.__class__.__name__
    if cur_dataseet_name == 'HAR':
        aug = magnitude_warp(sample, sigma=0.1, knot=150)
    elif cur_dataseet_name == 'EEG':
        aug = magnitude_warp(sample, sigma=0.1, knot=30)
    elif cur_dataseet_name == 'FD':
        aug = magnitude_warp(sample, sigma=0.01, knot=150)
    else:
        raise NameError
    #############################################
    aug2 = magnitude_warp(sample, sigma=0.1, knot=30)

    return aug, aug2

def jitter(x, sigma = 0.1):
    if not isinstance(x, np.ndarray):
        x = x.cpu().numpy()
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma=0.1):
    if not isinstance(x, np.ndarray):
        x = x.cpu().numpy()
    factor = np.random.normal(loc=1., scale = sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])

    return np.concatenate((ai), axis=1)

def window_slice(x, reduce_ratio=0.9):
    if not isinstance(x, np.ndarray):
        x = x.cpu().numpy()
    x = x.transpose((0, 2, 1))
    target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1] - target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i, :, dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len),
                                       pat[starts[i]:ends[i], dim]).T
    ret = ret.transpose((0, 2, 1))
    return ret

def magnitude_warp(x, sigma=0.1, knot=150):
    from scipy.interpolate import CubicSpline
    if not isinstance(x, np.ndarray):
        x = x.cpu().numpy()
    x = x.transpose((0, 2, 1))
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array(
            [CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    ret = ret.transpose((0, 2, 1))
    return ret


def time_warp(x, sigma=0.1, knot=30):
    from scipy.interpolate import CubicSpline
    if not isinstance(x, np.ndarray):
        x = x.cpu().numpy()
    x = x.transpose((0, 2, 1))
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(orig_steps)
            scale = (x.shape[1] - 1) / time_warp[-1]
            ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]).T
    ret = ret.transpose((0, 2, 1))
    return ret

def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])
    if not isinstance(x, np.ndarray):
        x = x.cpu().numpy()
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == 'random':
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.array_split(orig_steps, num_segs[i])

            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0, warp]
        else:
            ret[i] = pat

    return torch.from_numpy(ret)

import math
def data_transform_masked4cl(sample, masking_ratio, lm, positive_nums=None, distribution='geometric'):
    """Masked time series in time dimension"""

    if positive_nums is None:
        positive_nums = math.ceil(1.5 / (1 - masking_ratio))

    sample = sample.permute(0, 2, 1)

    sample_repeat = sample.repeat(positive_nums, 1, 1)
    mask = noise_mask(sample_repeat, masking_ratio, lm, distribution=distribution)
    x_masked = mask.float() * sample_repeat

    return x_masked.permute(0, 2, 1), mask.permute(0, 2, 1)


def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked
    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (
            1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask


def noise_mask(X, masking_ratio=0.25, lm=3, distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)
    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        mask = geom_noise_mask_single(X.shape[0] * X.shape[1] * X.shape[2], lm, masking_ratio)
        mask = mask.reshape(X.shape[0], X.shape[1], X.shape[2])
    elif distribution == 'masked_tail':
        mask = np.ones(X.shape, dtype=bool)
        for m in range(X.shape[0]):  # feature dimension

            keep_mask = np.zeros_like(mask[m, :], dtype=bool)
            n = math.ceil(keep_mask.shape[1] * (1 - masking_ratio))
            keep_mask[:, :n] = True
            mask[m, :] = keep_mask  # time dimension
    elif distribution == 'masked_head':
        mask = np.ones(X.shape, dtype=bool)
        for m in range(X.shape[0]):  # feature dimension

            keep_mask = np.zeros_like(mask[m, :], dtype=bool)
            n = math.ceil(keep_mask.shape[1] * masking_ratio)
            keep_mask[:, n:] = True
            mask[m, :] = keep_mask  # time dimension
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                p=(1 - masking_ratio, masking_ratio))
    return torch.tensor(mask).to(X.device)