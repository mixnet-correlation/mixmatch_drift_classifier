'''
Calculate the ROC curve points for a particular results dataset.
'''

from multiprocessing import Pool, cpu_count, shared_memory
from genwins import *
from alignment import *
from dataman import DataManager
from common import *
from const import *
from model import *
from sklearn.metrics import roc_curve
from keras.models import load_model
import tensorflow as tf
import numpy as np
from scipy import sparse
import pandas as pd
from functools import partial
from pathlib import Path
from tqdm import tqdm
import random
import argparse
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


n_cpus = cpu_count()


# Remove all nondeterminism
seed = int(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Limit GPU memorgy growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def config_parser():
    '''Define expected and possible command-line arguments.'''

    parser = argparse.ArgumentParser('Evaluate on test set.')

    parser.add_argument(
        'respath',
        type=str,
        help='path to results.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size.')
    parser.add_argument(
        '--resp-flows',
        type=int,
        default=5250,
        help='Resp flows.')
    parser.add_argument(
        '--two2one',
        action='store_true',
        help='Append flag for \'2 initiators to 1 responder\' setting.')

    return parser


def main(respath: str = './results/latest/',
         batch_size: int = 16,
         resp_flows: int = 5250,
         two2one: bool = False):
    '''Calculate the ROC curve points for the specified results.'''

    # Load datasets
    respath = Path(respath)

    # replace results file
    results_file = respath / 'step_results.csv'
    if results_file.exists():
        results_file.unlink(missing_ok=True)

    end = NWINS + 1
    if two2one:
        end = 1

    for win_idx in tqdm(range(end), total=end):

        # Load scores for current number of considered windows from
        # CSV files created using 'get_scores.py'.
        scores = pd.read_csv(respath / f'scores_{win_idx}.csv',
                             names=['data', 'acks', 'sums'])

        batches = len(scores['data']) / (batch_size * resp_flows)
        init_flows = int(batch_size * batches)

        # print(f"Before: {resp_flows=} (want: 5247), {init_flows=} (want: 5249)")
        if two2one:
            init_flows = resp_flows - 1
            resp_flows = ((resp_flows // batch_size) * batch_size) - 1
            assert init_flows == 5249
            assert resp_flows == 5247
        # print(f"After: {resp_flows=} (want: 5247), {init_flows=} (want: 5249)")

        full_size = init_flows * resp_flows

        positives = init_flows
        if two2one:
            positives = resp_flows

        negatives = full_size - positives

        # print(f"positives (want: 5,247): {positives=:,}")
        # print(f"negatives (want: 27,536,256): {negatives=:,}")
        # print(f"full_size (want: 27,541,503): {full_size=:,}")

        if two2one:
            assert positives == 5247
            assert negatives == 27536256
            assert full_size == 27541503

        # y_true represents the matrix of each considered initiator paired
        # with each responder. It has 1s on the diagonal, indicating matched
        # flow pairs. Every other cell in a row is 0, indicating an unmatched
        # flow pair. It's reshaped from matrix form to vector form.
        y_true = np.eye(init_flows,
                        resp_flows,
                        k=0,
                        dtype=np.int8).reshape(full_size)

        if two2one:
            y_true = np.eye(resp_flows,
                            init_flows,
                            k=0,
                            dtype=np.int8).reshape(full_size)

            # Positive i (zero-indexed) is at index: i * 5250.
            # print(f"{y_true.shape=}\n{y_true=}\n")
            # print(f"{y_true[0]=}")
            # print(f"{y_true[5250]=}")
            # print(f"{y_true[10500]=}")
            # print(f"{y_true[15750]=}")
            # print(f"{y_true[21000]=}")
            # print(f"{y_true[26250]=}")

            assert y_true[0] == 1
            assert y_true[5250] == 1
            assert y_true[10500] == 1
            assert y_true[15750] == 1
            assert y_true[21000] == 1
            assert y_true[26250] == 1

        # Sample by default 500 points from each respective array.
        data_ths = sample_ths(scores.data)
        acks_ths = sample_ths(scores.acks)
        sums_ths = sample_ths(scores.sums)

        for th_data, th_acks, th_sums in zip(data_ths, acks_ths, sums_ths):

            # print(f'{scores.data=}, {th_data=}')
            res_data = calculate_metrics(
                (scores.data > th_data).astype(int), y_true)
            res_acks = calculate_metrics(
                (scores.acks > th_acks).astype(int), y_true)
            res_both = calculate_metrics(
                (scores.sums > th_sums).astype(int), y_true)

            # write results to file
            with open(results_file, 'a', encoding='utf-8') as fo:

                if win_idx == NWINS + 1:
                    win_idx = 'all'

                print(
                    f'{win_idx},{th_data},drift,data,{res_data[0] / positives},{res_data[1] / negatives}',
                    file=fo)
                print(
                    f'{win_idx},{th_acks},drift,acks,{res_acks[0] / positives},{res_acks[1] / negatives}',
                    file=fo)
                print(
                    f'{win_idx},{th_sums},drift,both,{res_both[0] / positives},{res_both[1] / negatives}',
                    file=fo)
                fo.flush()


def calculate_metrics(y_pred, y_true):
    '''Calculate True Positives and False Positives of 'y_pred' based on 'y_true'.'''

    tps = ((y_true == 1) & (y_pred == 1)).astype(int).sum()
    fps = ((y_true == 0) & (y_pred == 1)).astype(int).sum()

    return tps, fps


if __name__ == '__main__':

    try:
        p = config_parser()
        args = p.parse_args()
        sys.exit(main(**vars(args)))

    except KeyboardInterrupt:
        sys.exit(1)
