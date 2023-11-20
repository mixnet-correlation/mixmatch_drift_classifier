'''
'''

import numpy as np
import common


def delay_sequences(s1, s2, delta=0):
    '''Aligns the single pair s1 and s2 according to the supplied delta value.
    Mind that delta can be positive or negative, depending on which direction
    (ingress, egress) needs to be shifted.'''

    # recast
    delta = np.int32(delta)
    l1, l2 = s1.shape[0], s2.shape[0]
    if delta >= 0:
        if l1 <= l2:
            if l1 + delta > l2:
                delayed_s1 = s1[:l2 - delta]
                delayed_s2 = s2[delta:]
            else:
                delayed_s1 = s1
                delayed_s2 = s2[delta:l1 + delta]
        else:
            delayed_s1 = s1[:l2 - delta]
            delayed_s2 = s2[delta:]
    else:
        if l1 > l2:
            if l1 + delta <= l2:
                delayed_s1 = s1[-delta:]
                delayed_s2 = s2[:l1 + delta]
            else:
                delayed_s1 = s1[-delta:l2 - delta]
                delayed_s2 = s2
        else:
            delayed_s1 = s1[-delta:]
            delayed_s2 = s2[:l1 + delta]

    return delayed_s1, delayed_s2


def align_pairs_channel(S1, S2, delays):
    '''Aligns either the data or the ack component of many pairs.'''

    m = delays.shape[1]
    # print(f'{n=}, {m=}, {S1.shape=}, {S2.shape=}, {delays.shape=}')

    D1, D2 = np.full(S1.shape, np.nan), np.full(S2.shape, np.nan)

    for i, (s1, s2) in enumerate(zip(S1, S2)):

        d1, d2 = delay_sequences(s1[~np.isnan(s1)],
                                 s2[~np.isnan(s2)],
                                 delays[i // m, i % m])

        D1[i, :len(d1)] = d1
        D2[i, :len(d2)] = d2

    return D1, D2


def align_pairs(S1, S2, delays):
    '''Aligns many pairs according to the supplied delays matrix.'''

    D1_data, D2_data = align_pairs_channel(S1[..., 0],
                                           S2[..., 0],
                                           delays[..., 0])

    D1_acks, D2_acks = align_pairs_channel(S1[..., 1],
                                           S2[..., 1],
                                           delays[..., 1])

    return common.stack(D1_data, D1_acks), common.stack(D2_data, D2_acks)


def align_single_pair(s1, s2, delay):
    '''Aligns a single pair cf. many as 'align_pairs()'.'''

    # Extract the data component.
    s1_data = s1[..., 0]
    s2_data = s2[..., 0]
    delay_data = delay[..., 0]

    # Prepare correctly-sized np.nan arrays.
    d1_data = np.full(s1_data.shape, np.nan)
    d2_data = np.full(s2_data.shape, np.nan)

    # Align the data component.
    d1_data_delayed, d2_data_delayed = delay_sequences(
        s1_data[~np.isnan(s1_data)], s2_data[~np.isnan(s2_data)], delay_data)

    # Overwrite the respective part in the prepared np.nan array.
    d1_data[:len(d1_data_delayed)] = d1_data_delayed
    d2_data[:len(d2_data_delayed)] = d2_data_delayed

    # Extract the ack component.
    s1_ack = s1[..., 1]
    s2_ack = s2[..., 1]
    delay_ack = delay[..., 1]

    # Prepare correctly-sized np.nan arrays.
    d1_ack = np.full(s1_ack.shape, np.nan)
    d2_ack = np.full(s2_ack.shape, np.nan)

    # Align the ack component.
    d1_ack_delayed, d2_ack_delayed = delay_sequences(
        s1_ack[~np.isnan(s1_ack)], s2_ack[~np.isnan(s2_ack)], delay_ack)

    # Overwrite the respective part in the prepared np.nan array.
    d1_ack[:len(d1_ack_delayed)] = d1_ack_delayed
    d2_ack[:len(d2_ack_delayed)] = d2_ack_delayed

    return common.stack(d1_data, d1_ack), common.stack(d2_data, d2_ack)
