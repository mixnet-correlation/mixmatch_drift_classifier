'''
Train our drift model.
'''

import os
import sys
import time
import shutil
import random
import argparse
import traceback
import logging
import logging.config
from pathlib import Path
import const
import common
import genwins
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from dataman import DataManager
import model as drift_model
from tensorflow.keras import optimizers

# from common import *
# import alignment
# from alignment import *
# import tensorflow.keras.backend as kb


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

log_file_path = const.ROOTPATH / 'logging.conf'
logging.config.fileConfig(log_file_path)
logger = logging.getLogger('train')

log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)
fh = logging.FileHandler('tensorflow.log')
fh.setLevel(logging.DEBUG)
log.addHandler(fh)

# Remove all nondeterminism
SEED = int(const.SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(const.SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Limit GPU memorgy growth
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


class ModelTrainer():
    '''Class to train our drift model.'''

    def __init__(
            self,
            model,
            triplet_generator,
            batch_size=16,
            learning_rate=1e-3):
        '''Initialize a class instance.'''

        self.model = model
        self.tg = triplet_generator
        self.batch_size = batch_size

        # self.opt = optimizers.Nadam(
        #     learning_rate=learning_rate,
        #     clipnorm=1.,
        #     clipvalue=1.,
        #     beta_1=0.9,
        #     beta_2=0.999)

        self.opt = optimizers.SGD(
            learning_rate=learning_rate,
            clipnorm=1.,
            clipvalue=1.,
            decay=1e-6,
            momentum=0.9,
            nesterov=True)

    def train_batch(self, data, backward=True):
        '''Batch-wise train on supplied dataset, possibly with weight updating.'''

        batch_count = data.init_size // self.batch_size
        final_count = batch_count * self.batch_size
        batch_iterator = enumerate(range(0, final_count, self.batch_size))
        logger.debug(
            f"Batch size: {self.batch_size}; Total number of batches: {batch_count}")
        logger.debug(f"Optimizer: {self.opt=}")

        for _, start in (pbar := tqdm(batch_iterator, total=batch_count)):

            end = start + self.batch_size

            # Select batch samples
            chunk = data.chunk(start, end)

            for channel in const.CHANNELS:

                # Get windows
                anc_wins, pns_wins = self.tg.get_windows(chunk.to_gateway,
                                                         chunk.from_gateway,
                                                         chunk.delays,
                                                         False,
                                                         False,
                                                         0)
                # labels
                labels = common.get_labels(self.tg)

                # Perform forward pass of both payload and ack traffic flows
                with tf.GradientTape() as dt:

                    # feed to model and get loss
                    pred = self.model((anc_wins[..., channel, None],
                                       pns_wins[..., channel, None]))
                    loss = common.cross_entropy(pred, labels)
                    del anc_wins, pns_wins

                    yield loss
                    msg = f"Channel {channel}, loss: {loss:.6f}"
                    logger.debug(msg)
                    pbar.set_description(msg)

                    if backward:
                        if loss > 0:
                            weights = self.model.trainable_weights
                            grads = dt.gradient(loss, weights)
                            self.opt.apply_gradients(zip(grads, weights))


def train(
        datapath='./data/latest/',
        respath='./results/latest',
        window_shift=1.0,
        window_size=100,
        learning_rate=1e-3,
        batch_size=32,
        num_epochs=100,
        model_kernel_size=8,
        model_pool_size=8,
        start_from_epoch=100,
        patience=3,
        min_delta=1e-5):
    '''Train our drift model.'''

    # Load datasets
    respath = Path(respath)
    data_mgr = DataManager(datapath)

    logger.info("Loading datasets...")
    train = data_mgr.load('train')
    val = data_mgr.load('val')

    # Instantiate model
    model = drift_model.DriftModel(logger,
                                   model_kernel_size,
                                   model_pool_size)

    win_gen = genwins.WinGenerator(model,
                                   window_shift,
                                   window_size,
                                   batch_size)

    model_trainer = ModelTrainer(model,
                                 win_gen,
                                 batch_size,
                                 learning_rate)

    patience_local = patience
    vl_best_loss_global = 1.0
    for epoch in range(num_epochs):
        tr_best_loss, vl_best_loss, = 1.0, 1.0

        start_time = time.time()
        logger.info(f"Epoch number {epoch}")

        # training
        train.shuffle()
        logger.info("Training...")
        for tr_loss in model_trainer.train_batch(train):
            if tr_loss < tr_best_loss:
                tr_best_loss = tr_loss

        # validation
        val.shuffle()
        logger.info("Validating...")
        for vl_loss in model_trainer.train_batch(val, backward=False):
            if vl_loss < vl_best_loss:
                vl_best_loss = vl_loss
                logger.debug(f'Saving model weights to {respath}')
                model_trainer.model.save(
                    respath / 'model.tf', save_format='tf')

        # Check early stopping
        delta = float(vl_best_loss_global - vl_best_loss)
        if delta < min_delta:
            logger.info("Val loss has not improved!")
            if epoch > start_from_epoch:
                patience_local -= 1
                if patience_local == 0:
                    logger.info(
                        f"Val loss did not improve for {patience} epochs: stop!")
                    return 0
        else:
            logger.info(f"Val loss has improved: {delta=}")
            patience_local = patience
        vl_best_loss_global = vl_best_loss

        logger.info(f"Best training loss: {tr_best_loss:.6f}, "
                    f"Best valdation loss: {vl_best_loss:.6f}")
        logger.info("Time taken: %.2fs" % (time.time() - start_time))

    return 0


def config_parser():
    '''Define expected and possible command-line arguments.'''

    parser = argparse.ArgumentParser('Train embeddings model.')
    parser.add_argument(
        '--datapath',
        type=str,
        default='./data/latest/',
        help='Path to input data.')
    parser.add_argument(
        '--respath',
        type=str,
        help='Path to results.')
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
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size.')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=100,
        help='Number of epochs.')
    parser.add_argument(
        '--model_kernel_size',
        type=int,
        default=8,
        help='Size of the convolutional kernels in each Conv1D kernel of drift.')
    parser.add_argument(
        '--model_pool_size',
        type=int,
        default=8,
        help='Size of pool to average in drift model.')

    return parser


def main(*args, **kwargs):
    '''Train our drift model.'''

    # set paths
    datapath = Path(kwargs['datapath'])
    if datapath.is_symlink():
        datapath = datapath.readlink()
    if 'respath' not in kwargs or kwargs['respath'] is None:
        kwargs['respath'] = const.RESPATH / datapath.name
    else:
        if os.path.isdir(kwargs['respath']):
            shutil.rmtree(kwargs['respath'])
    kwargs['respath'] = Path(kwargs['respath'])
    kwargs['respath'].mkdir(parents=True, exist_ok=True)

    # get experiment name
    with open(datapath / 'experiment.info', encoding='utf-8') as exp_fp:

        line = exp_fp.readline().strip()
        while not line.startswith("exp_num="):
            line = exp_fp.readline().strip()

        # Extract experiment ID from line that starts with "exp_num=".
        experiment = line.split("=")[1].split(",")[0]

    # set logger
    formatter = logging.Formatter(const.LOG_FORMAT)
    file_handler = logging.FileHandler(kwargs['respath'] / 'train.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(
        f"Started training of Experiment {experiment} with args: {kwargs=}")

    # create latest symlink
    (const.RESPATH / 'latest').unlink(missing_ok=True)
    (const.RESPATH / 'latest').symlink_to(kwargs['respath'])

    return sys.exit(train(**kwargs))


if __name__ == '__main__':

    parser = config_parser()
    args = parser.parse_args()

    try:
        sys.exit(main(None, **vars(args)))
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception:
        print(traceback.format_exc())
