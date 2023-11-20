'''
'''

from pathlib import Path
import numpy as np

from const import *


class DataManager():
    ''''''

    def __init__(self,
                 datapath):

        self.datapath = Path(datapath)

    def load(self,
             dataset):
        ''''''

        # Returns:
        #   initiator_from_gateway => (num_flowpairs, num_items_each_flowpair, 2)
        #   initiator_to_gateway => (num_flowpairs, num_items_each_flowpair, 2)
        #   responder_from_gateway => (num_flowpairs, num_items_each_flowpair, 2)
        #   responder_to_gateway => (num_flowpairs, num_items_each_flowpair, 2)
        data = np.load(self.datapath / f'{dataset}.npz')

        # Returns:
        #   DATASETSUBSET_delay_matrix => (num_flowpairs, num_flowpairs, 2)
        # e.g., for 'test':
        #   test_delay_matrix (5250, 5250, 2)
        delays = np.load(self.datapath / f"{dataset}_delay_matrix.npz")
        delays = delays[f'{dataset}_delay_matrix']

        return Dataset(data, delays)


class Dataset():
    ''''''

    def __init__(self,
                 data,
                 delays):

        self.data = {}
        for orig in ORIGINS:
            for direc in DIRECTIONS:
                self.data[f'{orig}_{direc}'] = data[f'{orig}_{direc}']

        # numpy_array.shape[0] returns the number of rows of numpy_array.
        self.init_size = data['initiator_to_gateway'].shape[0]
        self.resp_size = data['responder_to_gateway'].shape[0]

        self.delays = delays

    @property
    def initiator(self):
        ''''''

        return (self.data[f'initiator_{direc}'] for direc in DIRECTIONS)

    @property
    def responder(self):
        ''''''

        return (self.data[f'responder_{direc}'] for direc in DIRECTIONS)

    @property
    def to_gateway(self):
        '''Returns a generator per origin, containing the traffic data
        in direction towards the gateway.'''

        return (self.data[f'{orig}_to_gateway'] for orig in ORIGINS)

    @property
    def from_gateway(self):
        '''Returns a generator per origin, containing the traffic data
        in direction from the gateway.'''

        return (self.data[f'{orig}_from_gateway'] for orig in ORIGINS)

    def shuffle(self):
        ''''''

        perm = np.arange(self.init_size)
        np.random.shuffle(perm)

        for orig in ORIGINS:
            for direc in DIRECTIONS:
                self.data[f'{orig}_{direc}'] = self.data[f'{orig}_{direc}'][perm]

        self.delays = self.delays[perm, :][:, perm]

    def chunk(self,
              init_start: int,
              init_stop: int,
              resp_start: int = None,
              resp_stop: int = None,
              two2one: bool = False,
              final_batch: bool = False):
        '''Returns a subset of the total dataset, specified by
        ['init_start', 'init_stop') and ['resp_start', 'resp_stop').
        If we're in either case of scenario `two2one` and not in the
        final batch, instead return ['resp_start', 'resp_stop'] for
        the responder slice.'''

        if resp_start is None:
            resp_start = init_start

        if resp_stop is None:
            resp_stop = init_stop

        if two2one and not final_batch:
            resp_stop += 1

        chunk_data = {}
        for direc in DIRECTIONS:
            chunk_data[f'initiator_{direc}'] = self.data[f'initiator_{direc}'][init_start:init_stop]
            chunk_data[f'responder_{direc}'] = self.data[f'responder_{direc}'][resp_start:resp_stop]

        chunk_delays = \
            self.delays[init_start:init_stop, resp_start:resp_stop, :]

        return Dataset(chunk_data, chunk_delays)
