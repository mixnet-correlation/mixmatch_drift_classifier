'''
'''

import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import alignment
# from alignment import align_single_pair
from tensorflow.keras.backend import epsilon
from tensorflow.math import multiply, log, reduce_mean
from numpy.lib.stride_tricks import as_strided

EPS = epsilon()


def windowize(arr, wsize, addn, axis=1):
    step = int(wsize * addn)
    return rolling_window(arr, wsize, step, axis=1)


def cutoff_all(arr, index=-1):
    empty = np.isnan(arr[:, :, -1, ...]).any(axis=0)
    empty = empty[:, 0, ...] | empty[:, 1, ...]
    if len(empty.shape) > 1:
        empty = empty.any(axis=1)
    return arr[:, ~empty, ...]


def unstack(arr, axis=-1):
    return np.moveaxis(arr, axis, 0)


def stack(arr1, arr2, axis=-1):
    return np.stack((arr1, arr2), axis=axis)


def discard_along_axes(anc, pns):
    return unstack(cutoff_all(stack(anc, pns)))


def flatten_windows(arr):
    new_shape = (arr.shape[0] * arr.shape[1],) + arr.shape[2:]
    return arr.reshape(new_shape)


def window_view(arr, num_wins):
    new_shape = (arr.shape[0] // num_wins, num_wins,) + arr.shape[1:]
    return arr.reshape(new_shape)


def repeat(arr, size=None):
    '''Repeats each row of 'arr' a specified number of times
    into a new axis and reshapes the resulting array.'''

    if (size is None) or (size == 1):
        return arr

    # Duplicate each 'arr' row 'size' number of times
    # into a new axis. Reshape back into original form.
    rep = np.repeat(arr[:, None, ...], repeats=size, axis=1)
    new_shape = (rep.shape[0] * rep.shape[1],) + arr.shape[1:]

    # print(f"repeat: {rep.shape=}, {new_shape=}\n")

    return rep.reshape(new_shape)


def tile(arr, size=None):
    '''Duplicates the supplied 'arr' array a 'size'-based
    number of times into a new axis, and reshape the resulting
    array back into the original form.'''

    if (size is None) or (size == 1):
        return arr

    repeats = (size,) + (1,) * arr.ndim

    # Duplicate 'arr' array 'repeats' number of times
    # into a new axis. Reshape into new form.
    til = np.tile(arr[None, ...], repeats)
    new_shape = (til.shape[0] * til.shape[1],) + arr.shape[1:]

    # print(f"tile: {repeats=}, {til.shape=}, {new_shape=}\n")

    return til.reshape(new_shape)


def concatenate_pairwise(init,
                         resp):
    '''Builds cartesian product of 'init' and 'resp', through
    duplicating 'init' rows for the number of rows in 'resp'
    and duplicating the entire 'resp' array for the number
    of rows in 'init'. After merging the two appropriately
    duplicated arrays along the column axis, sorts each row's
    items in ascending order.'''

    return join(repeat(init, resp.shape[0]),
                tile(resp, init.shape[0]))


def gen_dataset_two2one_case1(init_to,
                              init_from,
                              resp_to,
                              resp_from,
                              delays,
                              shift_init: int):
    '''Returns newly constructed (simulated) flowpairs from the
    supplied 'init' and 'resp' traces, representing the case of
    `two2one` where flowpairs are either
        "2 matched initiators for the 1 (merged) responder"
    or
        "1 matched, 1 unmatched initiator for the 1 (merged) responder".
    Takes care of aligning the merged flows already (cf. concatenate_pairwise).'''

    assert init_to.shape == init_from.shape
    assert resp_to.shape == resp_from.shape

    # The returned arrays are of shape:
    #   rows: (#resp - 1) * (#init - 1)   => simulated pairings
    #   columns: 2 * (#timestamps_resp + #timestamps_init)   => 4 merged flows
    #   cells: 2   => data/ack
    case1_dataset_to = np.full((
        ((resp_to.shape[0] - 1) * (init_to.shape[0] - 1)),
        (2 * (resp_to.shape[1] + init_to.shape[1])),
        2),
        np.nan)
    case1_dataset_from = np.full((
        ((resp_from.shape[0] - 1) * (init_from.shape[0] - 1)),
        (2 * (resp_from.shape[1] + init_from.shape[1])),
        2),
        np.nan)

    # If indicated that we need to shift the initiator and delays
    # arrays (because we are processing the dataset in batches and
    # this is the non-first batch), do so via numpy.roll.
    if shift_init > 0:
        init_to = np.roll(init_to, -shift_init, axis=0)
        init_from = np.roll(init_from, -shift_init, axis=0)
        delays = np.roll(delays, -shift_init, axis=0)

    idx = 0

    for i in range((resp_to.shape[0] - 1)):

        k = 0
        shift_idx_0 = idx

        for _ in range((init_to.shape[0] - 1)):

            if k == i:
                k += 1

            # Merge first initiator-responder pair (always matched).
            first_merged_to = join(init_to[i], resp_to[i], axis=0)
            first_merged_from = join(init_from[i], resp_from[i], axis=0)

            # Align first pair.
            first_merged_aligned_to, first_merged_aligned_from = alignment.align_single_pair(
                first_merged_to, first_merged_from, delays[i, i, :])

            # Merge second initiator-responder pair (mostly unmatched).
            second_merged_to = join(init_to[k], resp_to[(i + 1)], axis=0)
            second_merged_from = join(init_from[k], resp_from[(i + 1)], axis=0)

            # Align second pair.
            second_merged_aligned_to, second_merged_aligned_from = alignment.align_single_pair(
                second_merged_to, second_merged_from, delays[k, (i + 1), :])

            # Creates (#init - 2) negatives for each positive.
            # Positive indices: i, i + 1. Skipped: i == j.
            case1_dataset_to[idx] = join(first_merged_aligned_to,
                                         second_merged_aligned_to,
                                         axis=0)
            case1_dataset_from[idx] = join(first_merged_aligned_from,
                                           second_merged_aligned_from,
                                           axis=0)

            k += 1
            idx += 1

        # Reverse the shift from before the for loops, such that
        # the cells of this row are in the correct final order.
        if shift_init > 0:

            case1_dataset_to[shift_idx_0:idx] = np.roll(
                case1_dataset_to[shift_idx_0:idx], shift_init, axis=0)

            case1_dataset_from[shift_idx_0:idx] = np.roll(
                case1_dataset_from[shift_idx_0:idx], shift_init, axis=0)

    # Reverse the shift from before the for loops, such that
    # the cells of this row are in the correct final order.
    if shift_init > 0:
        delays = np.roll(delays, shift_init, axis=0)

    return case1_dataset_to, case1_dataset_from


def gen_dataset_two2one_case2(init_to,
                              init_from,
                              resp_to,
                              resp_from,
                              delays,
                              shift_init):
    '''Returns newly constructed (simulated) flowpairs from the
    supplied 'init' and 'resp' traces, representing the case of
    `two2one` where flowpairs are either
        "2 matched initiators for the 1 (merged) responder"
    or
        "2 unmatched initiators for the 1 (merged) responder".
    Takes care of aligning the merged flows already (cf. concatenate_pairwise).'''

    assert init_to.shape == init_from.shape
    assert resp_to.shape == resp_from.shape

    # The returned arrays are of shape:
    #   rows: (#resp - 1) * (#init - 1)   => simulated pairings
    #   columns: 2 * (#timestamps_resp + #timestamps_init)   => 4 merged flows
    #   cells: 2   => data/ack
    case2_dataset_to = np.full((
        ((resp_to.shape[0] - 1) * (init_to.shape[0] - 1)),
        (2 * (resp_to.shape[1] + init_to.shape[1])),
        2),
        np.nan)
    case2_dataset_from = np.full((
        ((resp_from.shape[0] - 1) * (init_from.shape[0] - 1)),
        (2 * (resp_from.shape[1] + init_from.shape[1])),
        2),
        np.nan)

    # If indicated that we need to shift the initiator and delays
    # arrays (because we are processing the dataset in batches and
    # this is the non-first batch), do so via numpy.roll.
    if shift_init > 0:
        init_to = np.roll(init_to, -shift_init, axis=0)
        init_from = np.roll(init_from, -shift_init, axis=0)
        delays = np.roll(delays, -shift_init, axis=0)

    idx = 0

    for i in range((resp_to.shape[0] - 1)):

        shift_idx_0 = idx

        for j in range((init_to.shape[0] - 1)):

            if i == j:
                first = i
                second = i + 1

            else:

                if i < j:
                    first = (j + 1) % init_to.shape[0]
                    second = (j + 2) % init_to.shape[0]
                elif i > j:
                    first = j % init_to.shape[0]
                    second = (j + 1) % init_to.shape[0]

                while first in (i, (i + 1)):
                    first = (first + 1) % init_to.shape[0]

                while second in (i, (i + 1)):
                    second = (second + 1) % init_to.shape[0]

            # Merge first initiator-responder pair (mostly unmatched).
            first_merged_to = join(init_to[first], resp_to[i], axis=0)
            first_merged_from = join(
                init_from[first], resp_from[i], axis=0)

            # Align first pair.
            first_merged_aligned_to, first_merged_aligned_from = alignment.align_single_pair(
                first_merged_to, first_merged_from, delays[first, i, :])

            # Merge second initiator-responder pair (mostly unmatched).
            second_merged_to = join(init_to[second], resp_to[(i + 1)], axis=0)
            second_merged_from = join(
                init_from[second], resp_from[(i + 1)], axis=0)

            # Align second pair.
            second_merged_aligned_to, second_merged_aligned_from = alignment.align_single_pair(
                second_merged_to, second_merged_from, delays[second, (i + 1), :])

            # Creates (#init - 2) negatives for each positive.
            # Positive indices: i, i + 1.
            case2_dataset_to[idx] = join(first_merged_aligned_to,
                                         second_merged_aligned_to,
                                         axis=0)
            case2_dataset_from[idx] = join(first_merged_aligned_from,
                                           second_merged_aligned_from,
                                           axis=0)

            idx += 1

        # Reverse the shift from before the for loops, such that
        # the cells of this row are in the correct final order.
        if shift_init > 0:

            case2_dataset_to[shift_idx_0:idx] = np.roll(
                case2_dataset_to[shift_idx_0:idx], shift_init, axis=0)

            case2_dataset_from[shift_idx_0:idx] = np.roll(
                case2_dataset_from[shift_idx_0:idx], shift_init, axis=0)

    # Reverse the shift from before the for loops, such that
    # the cells of this row are in the correct final order.
    if shift_init > 0:
        delays = np.roll(delays, shift_init, axis=0)

    return case2_dataset_to, case2_dataset_from


def rolling_window(arr, wsize, step, axis=1):
    # Note: skips the last window if size of array is not
    # a multiple of the size of the window.
    assert step > 0
    shape = list(arr.shape)
    overlap = arr.shape[axis]
    if step < wsize:
        overlap += 1 - wsize
    shape[axis] = overlap // step
    shape.insert(axis + 1, wsize)

    strides = list(arr.strides)
    strides[axis] = arr.strides[axis] * step
    strides.insert(axis + 1, arr.strides[axis])

    return as_strided(arr, shape, strides)


def pad(
        array: np.ndarray,
        target_length: int,
        value: float = 0,
        axis: int = 0) -> np.ndarray:
    '''Pads 'array' to 'target_length' with 'value', if it has fewer items.'''

    # Determine how many slots to add to the AFTER side of 'array'.
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array

    # For 'axis', prepare 'pad' of BEFORE=0 and AFTER='pad_size'.
    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    # Right-pad 'array' with 'pad_size' items of value 'value',
    # such that its total length equals target_length.
    return np.pad(
        array,
        pad_width=npad,
        mode='constant',
        constant_values=(
            value,
        ))


def diff(arr, axis=1):
    return np.diff(arr, axis=axis)


def join(a, b, axis=1):
    '''Concatenates the two supplied arrays 'a' and 'b' along the
    specified axis, and sorts their items along the same axis.'''

    return np.sort(np.concatenate((a, b), axis=axis),
                   axis=axis, kind='mergesort')


def convert_to_tensor(v, add_axis=True):
    if add_axis:
        v = v[..., None]
    return tf.convert_to_tensor(v)


def get_best_loss(f):
    best_loss = 1.0
    for line in open(f):
        if "Best" in line:
            loss = float(line.strip().split()[-1])
            if loss < best_loss:
                best_loss = loss
    return best_loss


# LOSS
################################################################
def cross_entropy(y_pred, y_true):
    old_pred = y_pred
    y_pred = tf.clip_by_value(y_pred, EPS, 1 - EPS) + EPS
    t0 = multiply(log(y_pred), y_true)
    t1 = multiply(1. - y_true, log(1. - y_pred))
    loss = -reduce_mean(t0 + t1)
    return loss


def get_labels(tg):
    pw_init = repeat(np.arange(tg.init_flows), tg.init_flows)
    pw_resp = tile(np.arange(tg.resp_flows), tg.resp_flows)
    labels = convert_to_tensor(
        repeat(
            (pw_init == pw_resp).astype(
                np.float32),
            tg.nwins))
    return labels


def sample_ths(arr, n_sample=500):
    '''Assumes the array is sorted.

    Adapted from: https://stackoverflow.com/a/50685454
    '''
    arr = np.array(arr)
    arr = np.sort(arr)
    ind = gen_log_space(arr.size, n_sample)
    ind = arr.size - ind - 1
    ind = ind[::-1]
    return arr[ind]


def gen_log_space(limit, n=500):
    '''From: https://stackoverflow.com/a/12421820'''
    result = [1]
    if n > 1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit) / result[-1]) ** (1. / (n - len(result)))
    while len(result) < n:
        next_value = result[-1] * ratio
        if next_value - result[-1] >= 1:
            result.append(next_value)
        else:
            result.append(result[-1] + 1)
            ratio = (float(limit) / result[-1]) ** (1. / (n - len(result)))
    return np.array(list(map(lambda x: round(x) - 1, result)), dtype=np.uint64)


def gen_hist(nbins):
    return np.zeros(nbins - 1, dtype='int32'), np.linspace(0., 1., nbins)

# OTHERS:
################################################################


def latest_paths(path, n=1):
    return sorted(Path(path).glob('*/'), key=os.path.getmtime)[-n]


def update_symlink(frompath, topath):
    (topath).unlink(missing_ok=True)
    (topath).symlink_to(frompath)


def latest_symlink(path):
    update_symlink(latest_paths(path), path / 'latest')
