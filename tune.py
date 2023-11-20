'''
Tune the hyperparameters of our drift model using the
Bayesian-based hyperopt framework.
'''

import sys
import time
import shutil
import argparse
import traceback
import logging
import logging.config
import multiprocessing as mp
from pathlib import Path
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
from const import *
from common import *
from train import main as train


# configure paths
TUNEDIR = ROOTPATH / 'tuning' / time.strftime("%Y-%m-%d_%H-%M-%S")
TUNEDIR.mkdir(parents=True, exist_ok=True)

# configure logging
logging.config.fileConfig("./logging.conf")
logger = logging.getLogger("tune")


def train_and_evaluate(args):

    try:

        # parse dataset and windowize
        datapath = Path(args['datapath']).resolve()
        tunedir = TUNEDIR / time.strftime("%Y-%m-%d_%H-%M-%S")
        tunedir.mkdir(parents=True, exist_ok=True)
        args['respath'] = str(tunedir)

        logger.info(f"Starting a new evaluation {datapath}: {args}")
        logger.info("Running training in a subprocess...")
        proc = mp.Process(target=train, kwargs=args)
        proc.start()

        timeout = 7200
        start = time.time()
        while time.time() - start <= timeout:
            if proc.is_alive():
                time.sleep(30)
            else:
                break
        else:
            logger.error("timed out, killing process")
            proc.terminate()
        proc.join()

        # get last loss
        logger.debug("Training finished... collecting loss results")
        best_val_loss = get_best_loss(tunedir / 'train.log')
        logger.info(f"Finished evaluation with val loss: {best_val_loss}")

        return {
            'loss': best_val_loss,
            'status': STATUS_OK,
        }

    except Exception as excep:

        logger.error(traceback.format_exc())
        logger.error(f"Exception: {excep=}")

        return {
            'loss': 1000,
            'status': STATUS_FAIL,
        }


def main():
    '''Tune our drift model using the Bayesian hyperopt framework.'''

    # Configure parser
    parser = argparse.ArgumentParser('Hyperparameter tuning.')
    parser.add_argument(
        '--datapath',
        type=str,
        default='./data/latest/',
        help='Path to data directory.')
    args = parser.parse_args()

    # create dir structure
    space = {
        'datapath': args.datapath,
        'num_epochs': 30,
        'patience': 3,
        'start_from_epoch': 5,
        'min_delta': 1e-5,
        'window_shift': 1.0,
        'window_size': 100,
        'batch_size': 16,
        'learning_rate': 0.001,
        'model_kernel_size': hp.choice('model_kernel_size', [4, 8, 16, 32, 48, 64]),
    }

    trials = Trials()
    best = fmin(
        train_and_evaluate,
        space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
    )

    logger.info(f'best: {best}')
    shutil.move('debug_tune.log', TUNEDIR / 'debug.log')


if __name__ == '__main__':

    try:
        sys.exit(main())

    except KeyboardInterrupt:
        sys.exit(1)
