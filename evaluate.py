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

    p = argparse.ArgumentParser('Evaluate on test set.')
    p.add_argument('datapath',
                   type=str,
                   help='path to data.')
    p.add_argument('respath',
                   type=str,
                   help='path to results.')
    p.add_argument('--window-shift',
                   type=float,
                   default=1.0,
                   help='Window shift percentage.')
    p.add_argument('--window-size',
                   type=int,
                   default=100,
                   help='Window size.')
    p.add_argument('--batch-size',
                   type=int,
                   default=16,
                   help='Batch size.')
    return p


def main(datapath='./data/latest/', respath='./results/latest/',
         window_shift=1.0, window_size=100, batch_size=32):
    # Parse arguments
    datapath = Path(datapath)
    respath = Path(respath)

    thresholds = np.linspace(0, 0.999, 500)

    # Load datasets
    respath = Path(respath)
    dm = DataManager(datapath)
    logger.info("Loading test data...")
    test = dm.load('test')

    # Load model
    model = load_model(respath / 'model.tf', compile=False)

    # replace results file
    results_file = respath / 'step_results.csv'
    if results_file.exists():
        results_file.unlink(missing_ok=True)

    tg = WinGenerator(
        model,
        window_shift=window_shift,
        window_size=window_size,
        batch_size=batch_size)
    batch_count = test.init_size // batch_size
    end = batch_count * batch_size
    for start in (pbar := tqdm(range(0, end, batch_size), total=batch_count)):
        stop = start + batch_size

        chunk = test.chunk(start, stop, 0, test.resp_size)

        # Get windows
        # TODO: use rolling window with stride=1
        with tf.device(f'/cpu:0'):
            anc_wins, pns_wins = tg.get_windows(chunk.to_gateway,
                                                chunk.from_gateway,
                                                chunk.delays,
                                                False,
                                                False,
                                                0)

            # scores
            score_data = tg.calculate_scores(
                anc_wins[..., 0], pns_wins[..., 0], False, False).numpy()
            score_acks = tg.calculate_scores(
                anc_wins[..., 1], pns_wins[..., 1], False, False).numpy()

            # Apply thresholds
            assert tg.init_flows == batch_size
            assert tg.resp_flows == test.resp_size
            y_true = np.eye(
                tg.init_flows,
                test.resp_size,
                k=start,
                dtype=np.int8).reshape(
                (tg.init_flows * test.resp_size))
            for i, th in enumerate(thresholds):
                for win_idx in range(1, tg.nwins + 1):
                    score_data_w = score_data[:win_idx].mean(
                        axis=0).reshape((batch_size * test.resp_size))
                    score_acks_w = score_acks[:win_idx].mean(
                        axis=0).reshape((batch_size * test.resp_size))

                    # calculate metrics for only
                    res_data = calculate_metrics(
                        (score_data_w > th).astype(int), y_true)
                    res_acks = calculate_metrics(
                        (score_acks_w > th).astype(int), y_true)
                    res_both = calculate_metrics(
                        ((score_data_w + score_acks_w) / 2 > th).astype(int), y_true)

                    TPR = float(res_data[0]) / float(tg.init_flows)
                    FPR = float(res_data[1]) / \
                        float(score_data_w.size - tg.init_flows)
                    msg = (
                        f'{TPR=:.6f} ({res_data[0]}/{tg.init_flows}), '
                        f'{FPR=:.6f} ({res_data[1]} / {score_data_w.size - tg.init_flows}) ')
                    pbar.set_description(msg)

                    # write results to file
                    with open(results_file, 'a') as fo:
                        print(
                            f'{win_idx},{th},drift,data,{res_data[0]},{res_data[1]}',
                            file=fo)
                        print(
                            f'{win_idx},{th},drift,acks,{res_acks[0]},{res_acks[1]}',
                            file=fo)
                        print(
                            f'{win_idx},{th},drift,both,{res_both[0]},{res_both[1]}',
                            file=fo)
                        fo.flush()


def calculate_metrics(y_pred, y_true):
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
