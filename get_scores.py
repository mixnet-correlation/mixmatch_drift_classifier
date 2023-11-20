'''
'''

import os
import sys
import random
import logging
import argparse
from pathlib import Path
from multiprocessing import cpu_count
import const
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from dataman import DataManager
from genwins import WinGenerator
from keras.models import load_model

# from alignment import *
# from common import *
# from model import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Obtain the number of CPUs.
n_cpus = cpu_count()


# Remove all nondeterminism
SEED = int(const.SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(const.SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Limit GPU memorgy growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def config_parser():
    '''Define expected and possible command-line arguments.'''

    parser = argparse.ArgumentParser('Evaluate on test set.')

    parser.add_argument(
        'datapath',
        type=str,
        help='path to data.')
    parser.add_argument(
        'respath',
        type=str,
        help='path to results.')
    parser.add_argument(
        '--window-shift',
        type=float,
        default=1.0,
        help='Window shift percentage.')
    parser.add_argument(
        '--window-size',
        type=int,
        default=100,
        help='Window size.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size.')

    scenario = parser.add_mutually_exclusive_group()
    scenario.add_argument(
        '--two2one_case1',
        action='store_true',
        help='Append flag for \'2 initiators to 1 responder, case (1): '
        '2 matched or 1 matched, 1 unmatched\' setting.')
    scenario.add_argument(
        '--two2one_case2',
        action='store_true',
        help='Append flag for \'2 initiators to 1 responder, case (2): '
        '2 matched or 2 unmatched\' setting.')

    return parser


def main(datapath: str = './data/latest/',
         respath: str = './results/latest/',
         window_shift: float = 1.0,
         window_size: int = 100,
         batch_size: int = 16,
         two2one_case1: bool = False,
         two2one_case2: bool = False):
    '''Evaluate model specified by argument 'respath' on the TESTING subset
    of the dataset specified by argument 'datapath'.'''

    logger = logging.getLogger('get_scores')

    # Parse arguments
    datapath = Path(datapath)
    respath = Path(respath)

    # Load datasets
    respath = Path(respath)
    data_mgr = DataManager(datapath)
    logger.info("Loading test data...")
    test = data_mgr.load('test')

    # Load model
    model = load_model(respath / 'model.tf', compile=False)

    # replace results file
    results_file = respath / 'step_results.csv'
    if results_file.exists():
        results_file.unlink(missing_ok=True)

    for win_idx in range(const.NWINS + 1):
        win_fpath = respath / f'scores_{win_idx}.csv'
        if win_fpath.exists():
            win_fpath.unlink(missing_ok=True)

    # window generator
    win_gen = WinGenerator(model,
                           window_shift=window_shift,
                           window_size=window_size,
                           batch_size=batch_size)

    # Determine number of batches based on size of the
    # TESTING subset and the size of a batch.
    batch_count = test.init_size // batch_size
    if two2one_case1 or two2one_case2:
        new_size = batch_size * (test.resp_size - 1)
    else:
        new_size = batch_size * test.resp_size
    end = batch_count * batch_size

    for i, start in enumerate(pbar := tqdm(
            range(0, end, batch_size), total=batch_count)):

        stop = start + batch_size

        # Establish the list of input data items from the whole dataset that
        # make up this chunk, depending on which scenario we are evaluating.
        if two2one_case1 or two2one_case2:
            # initiators: [0, num_initiators), responders: [start, stop].
            chunk = test.chunk(0,
                               test.init_size,
                               start,
                               stop,
                               True,
                               stop == end)
        else:
            # initiators: [start, stop), responders: [0, num_responders).
            chunk = test.chunk(start,
                               stop,
                               0,
                               test.resp_size,
                               False,
                               stop == end)

        # Build anchor and positive/negative pairs in windows. Take care
        # to build simulated dataset of "2 initiators to 1 responder" if
        # 'two2one_case1' or 'two2one_case2' is true.
        anc_wins, pns_wins = win_gen.get_windows(chunk.to_gateway,
                                                 chunk.from_gateway,
                                                 chunk.delays,
                                                 two2one_case1,
                                                 two2one_case2,
                                                 start)

        # scores
        pbar.set_description(f'Calculating scores for batch {i}...')
        with tf.device('/cpu:0'):
            score_data = win_gen.calculate_scores(
                anc_wins[..., 0], pns_wins[..., 0], two2one_case1, two2one_case2).numpy()
            score_acks = win_gen.calculate_scores(
                anc_wins[..., 1], pns_wins[..., 1], two2one_case1, two2one_case2).numpy()

        if (two2one_case1 or two2one_case2) and (stop == end):
            new_size = (batch_size - 1) * (test.resp_size - 1)

        wins_all = list(range(1, const.NWINS + 1)) + [win_gen.nwins + 1]

        # Under 'two-to-one', we only want to look at the first window.
        if two2one_case1 or two2one_case2:
            wins_all = list(range(1, 2))

        for j, win_idx in enumerate(wins_all):

            pbar.set_description(f'{win_idx=}')
            score_data_w = score_data[:win_idx].mean(axis=0).reshape(new_size)
            score_acks_w = score_acks[:win_idx].mean(axis=0).reshape(new_size)
            score_sums_w = (score_data_w + score_acks_w) / 2

            df_win = pd.DataFrame({
                'data': score_data_w,
                'acks': score_acks_w,
                'sums': score_sums_w
            })

            df_win.to_csv(respath / f'scores_{j}.csv',
                          index=False,
                          mode='a',
                          header=False)


if __name__ == '__main__':

    try:
        p = config_parser()
        args = p.parse_args()
        sys.exit(main(**vars(args)))

    except KeyboardInterrupt:
        sys.exit(1)
