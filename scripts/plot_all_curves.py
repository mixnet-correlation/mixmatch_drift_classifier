'''
Plot curves of an evaluated model.

Call like so:
  user@machine $   python3 plot_all_curves.py ./results ./data/<EXP_NAME>
'''

import sys
import argparse
import subprocess as sp
from pathlib import Path, PosixPath
from tqdm import tqdm
from common import read_csv_results, read_csv_losses, plot_loss, plot_roc

sys.path.insert(0, str(Path.cwd()))


def parse_args():
    '''Declare expected and available command-line arguments and parse them.'''

    parser = argparse.ArgumentParser('Plot curves for files in tuning path.')

    parser.add_argument('tuningpath', help='path to the tuning results.')
    parser.add_argument('datapath')
    parser.add_argument('--negatives', type=int, default=5250)
    parser.add_argument('--clean', default=False, action="store_true")

    args = parser.parse_args()

    return args


def main():
    '''Plot relevant performance curves of evaluated model.'''

    args = parse_args()
    tuningpath = Path(args.tuningpath)

    # for each losses csv file, print loss curves graph
    for f in tqdm(tuningpath.rglob('*')):

        if not f.name == 'loss.csv':
            continue

        tqdm.write(f'Process {f.parent.name}...')
        datapath = Path(args.datapath)
        respath = tuningpath / f.parent.name

        with open(respath / 'train.log', encoding='utf-8') as flog:
            params = flog.readline().strip().split('kwargs=')[-1]
            param_dict = eval(params)

        name = '{window_size}_{window_shift}_{batch_size}_{learning_rate}'.format_map(
            param_dict)
        loss_png_path = respath / f'{f.parent.name}_{name}_loss.png'
        roc_png_path = respath / f'{f.parent.name}_{name}_roc.png'

        if args.clean:

            loss_png_path.unlink(missing_ok=True)
            roc_png_path.unlink(missing_ok=True)
            (respath / 'step_results.csv').unlink(missing_ok=True)

        else:

            if not (datapath / 'test.npz').is_file():
                raise Exception("Test data not found!")

            # evaluate accuracy on test set
            print(datapath, respath)

            if not (respath / 'step_results.csv').is_file():

                result = sp.run(['python',
                                 'evaluate.py',
                                 datapath,
                                 respath,
                                 '--window-shift',
                                 str(param_dict['window_shift']),
                                 '--window-size',
                                 str(param_dict['window_size'])])

                try:
                    result.check_returncode()

                except sp.CalledProcessError:
                    continue

            # plot ROC curve
            perf = read_csv_results(
                f.parent / 'step_results.csv',
                param_dict['batch_size'],
                args.negatives)
            plot_roc(perf, roc_png_path)

            # print loss graph or clean
            losses = read_csv_losses(f)
            plot_loss(losses, loss_png_path)

    return 0


if __name__ == "__main__":

    try:
        sys.exit(main())

    except KeyboardInterrupt:
        sys.exit(1)
