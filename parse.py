'''
Prepare one experiment's dataset (format 'mixcorr') in order to
subsequently train and evaluate our deep learning models on it.

Sample invocation of this file:
  user@machine $   python3 parse.py /path/to/datasets/low-delay \
    --delaymatpath /path/to/delay_matrices/low-delay --experiment 5
'''

import sys
import time
import fileinput
import argparse as ap
from pathlib import Path
import const
import common
import numpy as np


def load_to_array(csv_path, num_pairs=35000, dtype=np.float64):
    '''Return an array from a file by padding to max capture length.'''

    pairs = []
    max_length = 0

    with open(csv_path, 'r', encoding='utf-8') as csv_fp:

        for i, line in enumerate(csv_fp):

            if i >= num_pairs:
                break

            times = np.array(line.strip().split(), dtype=dtype)

            length = len(times)
            if length > max_length:
                max_length = length

            pairs.append(times)

    arr = np.array([common.pad(pair, max_length, np.nan) for pair in pairs])

    return arr


def concatenate_parts(datapath, dataset, dtype='data'):
    '''
    Some metadata traces might be split into multiple files due to file size
    limitations on GitHub. Identify these splits and unite them again.
    '''

    # Identify all traces that are split across multiple files due to size.
    to_concatenate = [p.stem for p in datapath.glob(f'{dataset}_*_{dtype}.*')]

    for file in np.unique(to_concatenate):

        print(f"file='{file}'\n")

        # Check if parts have already been concatenated.
        if (datapath / file).is_file():
            continue

        # If not, concatenate them and print to new file.
        with open(datapath / file, 'w', encoding='utf-8') as out_fp:

            for line in fileinput.input(sorted(datapath.glob(f'{file}.*'))):
                out_fp.write(line)

            fileinput.close()


def create_ack_symlinks(datapath, dataset):
    '''
    For acks:
       - ack initiator_to-gateway = data initiator_from-gateway
       - ack responder_to-gateway = data responder_from-gateway

    We create symlinks to save space.
    '''

    for origin in const.ORIGINS:

        for fpath in datapath.glob(f'{dataset}_{origin}_to_gateway_data*'):

            fname = fpath.name
            new_fname = fname.replace('data', 'ack')
            fname = fname.replace('to_gateway', 'from_gateway')

            # check if symlink exists
            if (datapath / new_fname).is_symlink():
                break

            abspath = (datapath / fname).resolve()
            (datapath / new_fname).symlink_to(abspath)


def ensure_outpath(outpath):
    '''Ensure 'outpath' exists in the file system by creating it, if need be.'''

    if outpath is None:
        outpath = const.DATAPATH / time.strftime("%Y-%m-%d_%H-%M-%S")
        outpath.mkdir(parents=True, exist_ok=True)
        common.update_symlink(outpath, const.DATAPATH / 'latest')

    return Path(outpath)


def parse_all(
        datapath,
        delaymatpath=None,
        outpath=None,
        exp_num=None,
        factor=1):
    '''
    Prepare each split of one experiment's dataset for training and
    evaluating our deep learning classifier on it afterwards.
    '''

    # Prepare relevant paths.
    datapath = Path(datapath)

    if delaymatpath is None:
        delaymatpath = datapath
    else:
        delaymatpath = Path(delaymatpath)

    # Make sure 'outpath' exists on disk.
    outpath = ensure_outpath(outpath)
    with open(outpath / 'experiment.info', 'w', encoding='utf-8') as exp_info_fp:
        exp_info_fp.write(f'datapath={datapath},\n')
        exp_info_fp.write(f'delaymatpath={delaymatpath},\n')
        exp_info_fp.write(f'outpath={outpath},\n')
        exp_info_fp.write(f'exp_num={exp_num},\n')
        exp_info_fp.write(f'factor={factor}\n')

    for dataset_split in ['train', 'val', 'test']:

        # If metadata traces are split across multiple files,
        # unite them into a single file again by concatenating.
        concatenate_parts(datapath, dataset_split, dtype='data')
        concatenate_parts(datapath, dataset_split, dtype='ack')

        # For acks, create symlinks to respective data files.
        create_ack_symlinks(datapath, dataset_split)

        num_pairs = const.PAIRS[dataset_split] // factor

        merged = {}
        max_len = 0

        for orig in const.ORIGINS:

            for direc in const.DIRECTIONS:

                key = f'{dataset_split}_{orig}_{direc}'
                data = load_to_array(datapath / f'{key}_data', num_pairs)
                acks = load_to_array(datapath / f'{key}_ack', num_pairs)

                # Pad to the same size.
                if acks.shape[1] < data.shape[1]:
                    acks = common.pad(acks, data.shape[1], np.nan, axis=1)

                elif data.shape[1] < acks.shape[1]:
                    data = common.pad(data, acks.shape[1], np.nan, axis=1)

                if max_len < data.shape[1]:
                    max_len = data.shape[1]

                # Now that 'data' and 'ack' are of the same size, we can
                # stack() them together, i.e., fuse each data and ack time
                # of an endpoint in either direction from their two distinct
                # arrays ('data', 'ack') into one.
                merged[f'{orig}_{direc}'] = common.stack(data, acks)

                # Make sure we see 'data' at the expected location '[..., 0]'
                # in the created array 'merged'.
                assert np.allclose(
                    merged[f'{orig}_{direc}'][..., 0], data, equal_nan=True)

        # Pad each data-ack timestamp row of each endpoint-direction
        # identifier to the identified maximum number of measurements
        # 'max_len' observed across all runs.
        for key, value in merged.items():
            merged[key] = data = common.pad(value, max_len, np.nan, axis=1)
            print(f'{dataset_split}, {key}, {merged[key].shape=}')

        # Load respective delay matrix from file.
        delays_from_file = np.load(
            delaymatpath / f'{dataset_split}_delay_matrix.npz')

        # In case of dataset split 'val', an alternatively used key
        # might be 'validation', so account for that case.
        delays = delays_from_file.get(f'delay_matrix_{dataset_split}')
        if (delays is None) and (dataset_split == 'val'):
            delays = delays_from_file.get('delay_matrix_validation')

        # Copy respective chunk of the delay matrix.
        new_delays = {}
        new_delays[f'{dataset_split}_delay_matrix'] = delays[:num_pairs, :num_pairs, :]

        # Store newly prepared data in compressed form to disk. This allows to
        # disregard the original dataset repository for subsequent training and
        # evaluation, as the 'outpath' location is now completely self-contained.
        # For each dataset split, it contains the experiment data in the correctly
        # assembled format as well as the respective delay matrix.
        np.savez_compressed(
            outpath /
            f'{dataset_split}_delay_matrix.npz',
            **new_delays)
        np.savez_compressed(f'{outpath / dataset_split}.npz', **merged)


def config_parser():
    '''Define expected and possible command-line arguments.'''

    parser = ap.ArgumentParser('Parse CSV files and split into windows.')
    parser.add_argument(
        'datapath',
        type=str,
        help='Path to CSV files.')
    parser.add_argument(
        '--delaymatpath',
        type=str,
        default=None,
        help='Path to delay matrices of respective experiment.')
    parser.add_argument(
        '--experiment',
        type=int,
        default=1,
        choices=const.EXPERIMENTS,
        help='Number of experiment under analysis.')
    parser.add_argument(
        '--outpath',
        default=None,
        help='Path where the windowed traces will be stored.')
    parser.add_argument(
        '--factor',
        type=int,
        default=1,
        help='Fraction of dataset to consider.')

    return parser


def main():
    '''
    Parse files in results repository into format that
    can be analyzed by our deep learning model.
    '''

    parser = config_parser()
    args = parser.parse_args()

    parse_all(
        args.datapath,
        delaymatpath=args.delaymatpath,
        outpath=args.outpath,
        exp_num=str(args.experiment),
        factor=args.factor)


if __name__ == '__main__':

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
