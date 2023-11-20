'''
'''

import logging
import tensorflow as tf
from alignment import align_pairs
from tensorflow.experimental.numpy import moveaxis
from common import concatenate_pairwise, windowize, \
    gen_dataset_two2one_case1, gen_dataset_two2one_case2, \
    discard_along_axes, flatten_windows, convert_to_tensor


logger = logging.getLogger('train')


class WinGenerator():
    ''''''

    def __init__(self,
                 model,
                 window_shift=1.0,
                 window_size=100,
                 batch_size=32):

        self.model = model
        self.window_shift = window_shift
        self.window_size = window_size
        self.batch_size = batch_size

        self.pairs = None
        self.nwins = None
        self.init_flows = None
        self.resp_flows = None

    def get_windows(self,
                    to_pairs,
                    from_pairs,
                    delays,
                    two2one_case1: bool,
                    two2one_case2: bool,
                    shift_init: int):
        '''Builds flow windows to train/validate/test from the respective dataset
        subset by constructing anchor ('anc'), and positive and negative ('pns')
        pairs from the matched input flowpairs. Aligns, windowizes, cleans up, and
        flattens the resulting merged 'to' and 'from' flows.'''

        # Establish the number of respective items (initator, responder)
        # based on the shape of the delay matrix.
        self.init_flows = delays.shape[0]
        self.resp_flows = delays.shape[1]

        # Unpack the 2-tuples for towards_gateway and from_gateway
        # into their respective origin.
        init_to, resp_to = to_pairs
        init_from, resp_from = from_pairs

        if two2one_case1:
            # Create dataset from aligned merged flows.
            anc_pairs, pns_pairs = gen_dataset_two2one_case1(init_to,
                                                             init_from,
                                                             resp_to,
                                                             resp_from,
                                                             delays,
                                                             shift_init)

        elif two2one_case2:
            # Create dataset from aligned merged flows.
            anc_pairs, pns_pairs = gen_dataset_two2one_case2(init_to,
                                                             init_from,
                                                             resp_to,
                                                             resp_from,
                                                             delays,
                                                             shift_init)

        else:

            # pairwise combinations between initiator and responder
            anc_pairs = concatenate_pairwise(init_to, resp_to)
            pns_pairs = concatenate_pairwise(init_from, resp_from)

            # alignment
            anc_pairs, pns_pairs = align_pairs(anc_pairs, pns_pairs, delays)

        # windowize
        anc_wins = windowize(anc_pairs, self.window_size, self.window_shift)
        pns_wins = windowize(pns_pairs, self.window_size, self.window_shift)

        # discard empty windows across axes
        anc_wins, pns_wins = discard_along_axes(anc_wins, pns_wins)

        # If we're in the 'two-to-one' setting, we want to focus on only
        # using the first window. Thus, discard all windows beyond 0.
        if two2one_case1 or two2one_case2:
            print(f"Before: {anc_wins.shape=}, {pns_wins.shape=}")
            anc_wins = anc_wins[:, 0, ...]
            pns_wins = pns_wins[:, 0, ...]
            print(f"After: {anc_wins.shape=}, {pns_wins.shape=}")

        # get num flows and num windows
        self.pairs = anc_wins.shape[0]
        self.nwins = anc_wins.shape[1]

        if two2one_case1 or two2one_case2:

            assert (self.init_flows - 1) * (self.resp_flows - 1) == self.pairs, \
                f"{(self.init_flows - 1)=} * {(self.resp_flows - 1)=} != {self.pairs=}"

        else:

            assert self.init_flows * self.resp_flows == self.pairs, \
                f"{self.init_flows=} * {self.resp_flows=} != {self.pairs=}"

        # flatten the windows dimension
        anc_wins = flatten_windows(anc_wins)
        pns_wins = flatten_windows(pns_wins)

        return anc_wins, pns_wins

    def calculate_scores(self,
                         anc_wins,
                         pns_wins,
                         two2one_case1: bool,
                         two2one_case2: bool,
                         add_axis=True,
                         roll=True):
        '''Obtain the model's scores for the supplied anchor and
        positive-negative windows of the batch.'''

        # convert to tensor
        anc_tens = convert_to_tensor(anc_wins, add_axis=add_axis)
        pns_tens = convert_to_tensor(pns_wins, add_axis=add_axis)

        # calculate score
        scores = self.model((anc_tens, pns_tens))

        if two2one_case1 or two2one_case2:

            scores = tf.reshape(
                scores,
                shape=((self.init_flows - 1),
                       (self.resp_flows - 1),
                       self.nwins))

        else:

            scores = tf.reshape(
                scores,
                shape=(self.init_flows, self.resp_flows, self.nwins))

        # roll axis for a more intuitive order:
        #   initiator-responder-windows
        if roll:
            scores = moveaxis(scores, 2, 0)

        return scores
