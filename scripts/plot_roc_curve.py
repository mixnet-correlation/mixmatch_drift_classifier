'''
Plot the ROC curve of an evaluated model.
'''

import sys
import pathlib
import argparse
from common import read_csv_results, plot_roc

sys.path.append(pathlib.Path.cwd())


def parse_args():
    '''Declare expected and available command-line arguments and parse them.'''

    parser = argparse.ArgumentParser('Plot ROC curve from csv file.')

    parser.add_argument('results', help='path to the results csv file.')

    args = parser.parse_args()

    return args


def main():
    '''Plot the ROC curve from an evaluated model's CSV file.'''

    args = parse_args()

    # parse parameters in the log
    results_csv = pathlib.Path(args.results)

    perf = read_csv_results(results_csv)

    plot_roc(perf, results_csv.parent / 'roc.png')

    return 0


if __name__ == "__main__":

    try:
        sys.exit(main())

    except KeyboardInterrupt:
        sys.exit(1)
