'''
Test many of the foundational common functions.
'''

import unittest
import numpy as np
import numpy.testing as test

# from common import *
import common
from dataman import Dataset
from alignment import align_pairs


def create_test_dataset_small():
    '''Returns a 4x4 basic dataset for testing our dataset generation methods.'''

    data = {
        "initiator_to_gateway": np.arange(4).repeat(4).reshape((4, 2, 2)),
        "responder_to_gateway": np.arange(4).repeat(4).reshape((4, 2, 2)),
        "initiator_from_gateway": np.arange(10, 14).repeat(4).reshape((4, 2, 2)),
        "responder_from_gateway": np.arange(10, 14).repeat(4).reshape((4, 2, 2)),
    }

    delays = np.zeros((4, 4, 2), dtype = int)

    return Dataset(data, delays)


def create_test_dataset_medium():
    '''Returns a 6x6 basic dataset for testing our dataset generation methods.'''

    data = {
        "initiator_to_gateway": np.arange(6).repeat(4).reshape((6, 2, 2)),
        "responder_to_gateway": np.arange(6).repeat(4).reshape((6, 2, 2)),
        "initiator_from_gateway": np.arange(10, 16).repeat(4).reshape((6, 2, 2)),
        "responder_from_gateway": np.arange(10, 16).repeat(4).reshape((6, 2, 2)),
    }

    delays = np.zeros((6, 6, 2), dtype = int)

    return Dataset(data, delays)


def create_test_dataset_large():
    '''Returns a 9x9 basic dataset for testing our dataset generation methods.'''

    data = {
        "initiator_to_gateway": np.arange(9).repeat(4).reshape((9, 2, 2)),
        "responder_to_gateway": np.arange(9).repeat(4).reshape((9, 2, 2)),
        "initiator_from_gateway": np.arange(10, 19).repeat(4).reshape((9, 2, 2)),
        "responder_from_gateway": np.arange(10, 19).repeat(4).reshape((9, 2, 2)),
    }

    delays = np.zeros((9, 9, 2), dtype = int)

    return Dataset(data, delays)


def create_test_dataset_large_delays():
    '''Returns a 10x9 dataset with a non-zero delays matrix for testing our
    dataset generation methods with batching.'''

    data = {
        "initiator_to_gateway": np.arange(10).repeat(4).reshape((10, 2, 2)),
        "responder_to_gateway": np.arange(9).repeat(4).reshape((9, 2, 2)),
        "initiator_from_gateway": np.arange(10, 20).repeat(4).reshape((10, 2, 2)),
        "responder_from_gateway": np.arange(10, 19).repeat(4).reshape((9, 2, 2)),
    }

    delays = np.array([
        [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0], [-1, -1], [0, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [2, 2], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [-2, -2], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    ])

    return Dataset(data, delays)


def create_test_dataset_huge():
    '''Returns a 242x242 dataset for testing our dataset generation methods.'''

    data = {
        "initiator_to_gateway": np.random.rand(242, 50, 2),
        "responder_to_gateway": np.random.rand(242, 50, 2),
        "initiator_from_gateway": np.random.rand(242, 50, 2),
        "responder_from_gateway": np.random.rand(242, 50, 2),
    }

    delays = np.random.randint(low = -10, high = 10, size = (242, 242, 2), dtype = int)

    return Dataset(data, delays)


class TestArrayManipulationMethods(unittest.TestCase):

    def test_rolling_window(self):
        # step < wsize
        a = np.arange(3 * 10).reshape((3, 10))
        b = common.rolling_window(a, 4, 2)
        self.assertEqual(b.shape, (3, 3, 4))

        # step == wsize
        a = np.arange(3 * 10).reshape((3, 10))
        b = common.rolling_window(a, 4, 4)
        self.assertEqual(b.shape, (3, 2, 4))

        # 3 dimensions
        a = np.arange(3 * 10 * 2).reshape((3, 10, 2))
        b = common.rolling_window(a, 4, 4)
        self.assertEqual(b.shape, (3, 2, 4, 2))

        # concrete array no overlap
        a = np.array([[0, 1, 2, 3, 4],
                      [5, 6, 7, 8, 9]])
        expected = np.array([[[0, 1], [2, 3]],
                             [[5, 6], [7, 8]]])
        result = common.rolling_window(a, 2, 2)
        test.assert_array_equal(result, expected)

        # concrete array with overlap
        a = np.array([[0, 1, 2, 3, 4],
                      [5, 6, 7, 8, 9]])
        expected = np.array([[[0, 1], [1, 2], [2, 3], [3, 4]],
                             [[5, 6], [6, 7], [7, 8], [8, 9]]])
        result = common.rolling_window(a, 2, 1)
        test.assert_array_equal(result, expected)

    def test_windowize(self):
        # addn = 1
        a = np.arange(3 * 10).reshape((3, 10))
        result = common.windowize(a, 4, 1, axis=1)
        expected = common.rolling_window(a, 4, 4, axis=1)
        test.assert_array_equal(result, expected)

        # addn = 0.5
        a = np.arange(3 * 10).reshape((3, 10))
        result = common.windowize(a, 4, 0.5, axis=1)
        expected = common.rolling_window(a, 4, 2, axis=1)
        test.assert_array_equal(result, expected)

    def test_repeat(self):
        # concrete array
        a = np.array([[0, 1, 2, 3, 4],
                      [5, 6, 7, 8, 9]])

        expected = np.array([[0, 1, 2, 3, 4],
                             [0, 1, 2, 3, 4],

                             [5, 6, 7, 8, 9],
                             [5, 6, 7, 8, 9]])
        result = common.repeat(a, a.shape[0])
        test.assert_array_equal(result, expected)

        # 3 dimensions
        a = np.arange(3 * 10 * 2).reshape((3, 10, 2))
        b = common.repeat(a, a.shape[0])
        self.assertEqual(b.shape, (9, 10, 2))
        test.assert_array_equal(b[0, :, :], b[1, :, :])

    def test_tile(self):
        # concrete array
        a = np.array([[0, 1, 2, 3, 4],
                      [5, 6, 7, 8, 9]])

        expected = np.array([[0, 1, 2, 3, 4],
                             [5, 6, 7, 8, 9],

                             [0, 1, 2, 3, 4],
                             [5, 6, 7, 8, 9]])
        result = common.tile(a, a.shape[0])
        test.assert_array_equal(result, expected)

        # 3 dimensions
        a = np.arange(3 * 10 * 2).reshape((3, 10, 2))
        b = common.tile(a, a.shape[0])
        self.assertEqual(b.shape, (9, 10, 2))
        test.assert_array_equal(b[0, :, :], b[3, :, :])

    def test_join(self):
        a = np.array([[2, 5, 6, 8, np.nan],
                      [3, 7, 8, 8, 10]])
        b = np.array([[3, 4, 5, np.nan, np.nan],
                      [4, 5, 9, 10, np.nan]])
        c = common.join(a, b)
        exp = np.array([[2, 3, 4, 5, 5, 6, 8, np.nan, np.nan, np.nan],
                        [3, 4, 5, 7, 8, 8, 9, 10, 10, np.nan]])
        test.assert_array_equal(c, exp)

    def test_concatenate_pairwise(self):

        a = np.array([[2, 5, 6, 8, np.nan],
                      [3, 7, 8, 8, 10]])
        b = np.array([[3, 4, 5, np.nan, np.nan],
                      [4, 5, 9, 10, np.nan]])
        c = common.concatenate_pairwise(a, b)
        exp = np.array([[2, 3, 4, 5, 5, 6, 8, np.nan, np.nan, np.nan],
                        [2, 4, 5, 5, 6, 8, 9, 10, np.nan, np.nan],
                        [3, 3, 4, 5, 7, 8, 8, 10, np.nan, np.nan],
                        [3, 4, 5, 7, 8, 8, 9, 10, 10, np.nan],])
        test.assert_array_equal(c, exp)

        # different number of rows
        a = np.array([[2, 5, 6, 8, np.nan],])
        b = np.array([[3, 4, 5, np.nan, np.nan],
                      [4, 5, 9, 10, np.nan]])
        c = common.concatenate_pairwise(a, b)
        exp = np.array([[2, 3, 4, 5, 5, 6, 8, np.nan, np.nan, np.nan],
                        [2, 4, 5, 5, 6, 8, 9, 10, np.nan, np.nan]])
        test.assert_array_equal(c, exp)

        init = np.arange(8).repeat(2).reshape((4, 2, 2))
        resp = np.arange(8, 16).repeat(2).reshape((4, 2, 2))
        anchors_exp = np.array([
            [[0, 0], [1, 1], [8, 8], [9, 9]],
            [[0, 0], [1, 1], [10, 10], [11, 11]],
            [[0, 0], [1, 1], [12, 12], [13, 13]],
            [[0, 0], [1, 1], [14, 14], [15, 15]],
            [[2, 2], [3, 3], [8, 8], [9, 9]],
            [[2, 2], [3, 3], [10, 10], [11, 11]],
            [[2, 2], [3, 3], [12, 12], [13, 13]],
            [[2, 2], [3, 3], [14, 14], [15, 15]],
            [[4, 4], [5, 5], [8, 8], [9, 9]],
            [[4, 4], [5, 5], [10, 10], [11, 11]],
            [[4, 4], [5, 5], [12, 12], [13, 13]],
            [[4, 4], [5, 5], [14, 14], [15, 15]],
            [[6, 6], [7, 7], [8, 8], [9, 9]],
            [[6, 6], [7, 7], [10, 10], [11, 11]],
            [[6, 6], [7, 7], [12, 12], [13, 13]],
            [[6, 6], [7, 7], [14, 14], [15, 15]]
        ])

        anchors = common.concatenate_pairwise(init, resp)

        test.assert_array_equal(anchors, anchors_exp)

    def test_concatenate_pairwise_chunked(self):
        '''Tests how concatenate_pairwise() works chunked.'''

        print("\n")

        dataset = create_test_dataset_huge()

        batch_size = 3
        batch_count = dataset.init_size // batch_size
        end = batch_count * batch_size

        # Get the entire-dataset result.

        init_to_whole, resp_to_whole = dataset.to_gateway
        init_from_whole, resp_from_whole = dataset.from_gateway

        anc_pairs_whole = common.concatenate_pairwise(init_to_whole, resp_to_whole)
        pns_pairs_whole = common.concatenate_pairwise(init_from_whole, resp_from_whole)

        anc_pairs_whole, pns_pairs_whole = align_pairs(
            anc_pairs_whole, pns_pairs_whole, dataset.delays)

        # Get the batch-sized results.
        for i, start in enumerate(range(0, end, batch_size)):

            stop = start + batch_size
            print(f"Batch {i=}:  {start=} => {stop=}")

            chunk = dataset.chunk(
                start, stop, 0, dataset.resp_size, False, stop == end)

            init_to_chunk, resp_to_chunk = chunk.to_gateway
            init_from_chunk, resp_from_chunk = chunk.from_gateway

            anc_pairs_chunk = common.concatenate_pairwise(init_to_chunk, resp_to_chunk)
            pns_pairs_chunk = common.concatenate_pairwise(init_from_chunk, resp_from_chunk)

            anc_pairs_chunk, pns_pairs_chunk = align_pairs(
                anc_pairs_chunk, pns_pairs_chunk, chunk.delays)

            test.assert_array_equal(
                anc_pairs_chunk,
                anc_pairs_whole[(start * dataset.resp_size):(stop * dataset.resp_size), :, :])
            test.assert_array_equal(
                pns_pairs_chunk,
                pns_pairs_whole[(start * dataset.resp_size):(stop * dataset.resp_size), :, :])

        print()

    def test_gen_dataset_two2one_case1(self):
        '''Test for case 1 of the 'two-to-one' scenario.'''

        dataset = create_test_dataset_small()

        init_to, resp_to = dataset.to_gateway
        init_from, resp_from = dataset.from_gateway

        anc_pairs_exp = np.array([
            [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [1, 1], [1, 1]],
            [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [2, 2], [2, 2]],
            [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [3, 3], [3, 3]],
            [[0, 0], [0, 0], [1, 1], [1, 1], [1, 1], [1, 1], [2, 2], [2, 2]],
            [[1, 1], [1, 1], [1, 1], [1, 1], [2, 2], [2, 2], [2, 2], [2, 2]],
            [[1, 1], [1, 1], [1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]],
            [[0, 0], [0, 0], [2, 2], [2, 2], [2, 2], [2, 2], [3, 3], [3, 3]],
            [[1, 1], [1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [3, 3], [3, 3]],
            [[2, 2], [2, 2], [2, 2], [2, 2], [3, 3], [3, 3], [3, 3], [3, 3]]
        ])
        pns_pairs_exp = np.array([
            [[10, 10], [10, 10], [10, 10], [10, 10], [11, 11], [11, 11], [11, 11], [11, 11]],
            [[10, 10], [10, 10], [10, 10], [10, 10], [11, 11], [11, 11], [12, 12], [12, 12]],
            [[10, 10], [10, 10], [10, 10], [10, 10], [11, 11], [11, 11], [13, 13], [13, 13]],
            [[10, 10], [10, 10], [11, 11], [11, 11], [11, 11], [11, 11], [12, 12], [12, 12]],
            [[11, 11], [11, 11], [11, 11], [11, 11], [12, 12], [12, 12], [12, 12], [12, 12]],
            [[11, 11], [11, 11], [11, 11], [11, 11], [12, 12], [12, 12], [13, 13], [13, 13]],
            [[10, 10], [10, 10], [12, 12], [12, 12], [12, 12], [12, 12], [13, 13], [13, 13]],
            [[11, 11], [11, 11], [12, 12], [12, 12], [12, 12], [12, 12], [13, 13], [13, 13]],
            [[12, 12], [12, 12], [12, 12], [12, 12], [13, 13], [13, 13], [13, 13], [13, 13]]
        ])

        anc_pairs, pns_pairs = common.gen_dataset_two2one_case1(init_to,
                                                                init_from,
                                                                resp_to,
                                                                resp_from,
                                                                dataset.delays,
                                                                shift_init=0)

        test.assert_array_equal(anc_pairs, anc_pairs_exp)
        test.assert_array_equal(pns_pairs, pns_pairs_exp)

    def test_gen_dataset_two2one_case1_chunked(self):
        '''Tests how gen_dataset_two2one_case1() works chunked.'''

        print("\n")

        dataset = create_test_dataset_huge()

        batch_size = 3
        batch_count = dataset.init_size // batch_size
        end = batch_count * batch_size

        # Get the entire-dataset result.

        init_to_whole, resp_to_whole = dataset.to_gateway
        init_from_whole, resp_from_whole = dataset.from_gateway

        anc_pairs_whole, pns_pairs_whole = common.gen_dataset_two2one_case1(
            init_to_whole,
            init_from_whole,
            resp_to_whole,
            resp_from_whole,
            dataset.delays,
            shift_init=0)

        # Get the batch-sized results.
        for i, start in enumerate(range(0, end, batch_size)):

            stop = start + batch_size
            print(f"Batch {i=}:  {start=} => {stop=}")

            chunk = dataset.chunk(
                0, dataset.init_size, start, stop, True, stop == end)

            init_to_chunk, resp_to_chunk = chunk.to_gateway
            init_from_chunk, resp_from_chunk = chunk.from_gateway

            assert init_to_chunk.shape == (dataset.init_size, 50, 2)

            if stop != end:
                assert resp_to_chunk.shape == ((batch_size + 1), 50, 2)
            else:
                assert resp_to_chunk.shape == (batch_size, 50, 2)

            assert init_from_chunk.shape == (dataset.init_size, 50, 2)

            if stop != end:
                assert resp_from_chunk.shape == ((batch_size + 1), 50, 2)
            else:
                assert resp_from_chunk.shape == (batch_size, 50, 2)

            anc_pairs_chunk, pns_pairs_chunk = common.gen_dataset_two2one_case1(
                init_to_chunk,
                init_from_chunk,
                resp_to_chunk,
                resp_from_chunk,
                chunk.delays,
                shift_init=start)

            if stop != end:
                test.assert_array_equal(
                    anc_pairs_chunk,
                    anc_pairs_whole[(start * (dataset.init_size - 1)):(stop * (dataset.init_size - 1)), :, :])
                test.assert_array_equal(
                    pns_pairs_chunk,
                    pns_pairs_whole[(start * (dataset.init_size - 1)):(stop * (dataset.init_size - 1)), :, :])
            else:
                test.assert_array_equal(
                    anc_pairs_chunk,
                    anc_pairs_whole[(start * (dataset.init_size - 1)):((stop - 1) * (dataset.init_size - 1)), :, :])
                test.assert_array_equal(
                    pns_pairs_chunk,
                    pns_pairs_whole[(start * (dataset.init_size - 1)):((stop - 1) * (dataset.init_size - 1)), :, :])

        print()

    def test_gen_dataset_two2one_case2(self):
        '''Test for case 2 of the 'two-to-one' scenario.'''

        dataset = create_test_dataset_medium()

        init_to, resp_to = dataset.to_gateway
        init_from, resp_from = dataset.from_gateway

        anc_pairs_exp = np.array([
            [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [1, 1], [1, 1]],
            [[0, 0], [0, 0], [1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]],
            [[0, 0], [0, 0], [1, 1], [1, 1], [3, 3], [3, 3], [4, 4], [4, 4]],
            [[0, 0], [0, 0], [1, 1], [1, 1], [4, 4], [4, 4], [5, 5], [5, 5]],
            [[0, 0], [0, 0], [1, 1], [1, 1], [2, 2], [2, 2], [5, 5], [5, 5]],
            [[0, 0], [0, 0], [1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]],
            [[1, 1], [1, 1], [1, 1], [1, 1], [2, 2], [2, 2], [2, 2], [2, 2]],
            [[1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3], [4, 4], [4, 4]],
            [[1, 1], [1, 1], [2, 2], [2, 2], [4, 4], [4, 4], [5, 5], [5, 5]],
            [[0, 0], [0, 0], [1, 1], [1, 1], [2, 2], [2, 2], [5, 5], [5, 5]],
            [[0, 0], [0, 0], [1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]],
            [[1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3], [4, 4], [4, 4]],
            [[2, 2], [2, 2], [2, 2], [2, 2], [3, 3], [3, 3], [3, 3], [3, 3]],
            [[2, 2], [2, 2], [3, 3], [3, 3], [4, 4], [4, 4], [5, 5], [5, 5]],
            [[0, 0], [0, 0], [2, 2], [2, 2], [3, 3], [3, 3], [5, 5], [5, 5]],
            [[0, 0], [0, 0], [1, 1], [1, 1], [3, 3], [3, 3], [4, 4], [4, 4]],
            [[1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3], [4, 4], [4, 4]],
            [[2, 2], [2, 2], [3, 3], [3, 3], [4, 4], [4, 4], [5, 5], [5, 5]],
            [[3, 3], [3, 3], [3, 3], [3, 3], [4, 4], [4, 4], [4, 4], [4, 4]],
            [[0, 0], [0, 0], [3, 3], [3, 3], [4, 4], [4, 4], [5, 5], [5, 5]],
            [[0, 0], [0, 0], [1, 1], [1, 1], [4, 4], [4, 4], [5, 5], [5, 5]],
            [[1, 1], [1, 1], [2, 2], [2, 2], [4, 4], [4, 4], [5, 5], [5, 5]],
            [[2, 2], [2, 2], [3, 3], [3, 3], [4, 4], [4, 4], [5, 5], [5, 5]],
            [[0, 0], [0, 0], [3, 3], [3, 3], [4, 4], [4, 4], [5, 5], [5, 5]],
            [[4, 4], [4, 4], [4, 4], [4, 4], [5, 5], [5, 5], [5, 5], [5, 5]]
        ])
        pns_pairs_exp = np.array([
            [[10, 10], [10, 10], [10, 10], [10, 10], [11, 11], [11, 11], [11, 11], [11, 11]],
            [[10, 10], [10, 10], [11, 11], [11, 11], [12, 12], [12, 12], [13, 13], [13, 13]],
            [[10, 10], [10, 10], [11, 11], [11, 11], [13, 13], [13, 13], [14, 14], [14, 14]],
            [[10, 10], [10, 10], [11, 11], [11, 11], [14, 14], [14, 14], [15, 15], [15, 15]],
            [[10, 10], [10, 10], [11, 11], [11, 11], [12, 12], [12, 12], [15, 15], [15, 15]],
            [[10, 10], [10, 10], [11, 11], [11, 11], [12, 12], [12, 12], [13, 13], [13, 13]],
            [[11, 11], [11, 11], [11, 11], [11, 11], [12, 12], [12, 12], [12, 12], [12, 12]],
            [[11, 11], [11, 11], [12, 12], [12, 12], [13, 13], [13, 13], [14, 14], [14, 14]],
            [[11, 11], [11, 11], [12, 12], [12, 12], [14, 14], [14, 14], [15, 15], [15, 15]],
            [[10, 10], [10, 10], [11, 11], [11, 11], [12, 12], [12, 12], [15, 15], [15, 15]],
            [[10, 10], [10, 10], [11, 11], [11, 11], [12, 12], [12, 12], [13, 13], [13, 13]],
            [[11, 11], [11, 11], [12, 12], [12, 12], [13, 13], [13, 13], [14, 14], [14, 14]],
            [[12, 12], [12, 12], [12, 12], [12, 12], [13, 13], [13, 13], [13, 13], [13, 13]],
            [[12, 12], [12, 12], [13, 13], [13, 13], [14, 14], [14, 14], [15, 15], [15, 15]],
            [[10, 10], [10, 10], [12, 12], [12, 12], [13, 13], [13, 13], [15, 15], [15, 15]],
            [[10, 10], [10, 10], [11, 11], [11, 11], [13, 13], [13, 13], [14, 14], [14, 14]],
            [[11, 11], [11, 11], [12, 12], [12, 12], [13, 13], [13, 13], [14, 14], [14, 14]],
            [[12, 12], [12, 12], [13, 13], [13, 13], [14, 14], [14, 14], [15, 15], [15, 15]],
            [[13, 13], [13, 13], [13, 13], [13, 13], [14, 14], [14, 14], [14, 14], [14, 14]],
            [[10, 10], [10, 10], [13, 13], [13, 13], [14, 14], [14, 14], [15, 15], [15, 15]],
            [[10, 10], [10, 10], [11, 11], [11, 11], [14, 14], [14, 14], [15, 15], [15, 15]],
            [[11, 11], [11, 11], [12, 12], [12, 12], [14, 14], [14, 14], [15, 15], [15, 15]],
            [[12, 12], [12, 12], [13, 13], [13, 13], [14, 14], [14, 14], [15, 15], [15, 15]],
            [[10, 10], [10, 10], [13, 13], [13, 13], [14, 14], [14, 14], [15, 15], [15, 15]],
            [[14, 14], [14, 14], [14, 14], [14, 14], [15, 15], [15, 15], [15, 15], [15, 15]]
        ])

        anc_pairs, pns_pairs = common.gen_dataset_two2one_case2(init_to,
                                                                init_from,
                                                                resp_to,
                                                                resp_from,
                                                                dataset.delays,
                                                                shift_init=0)

        test.assert_array_equal(anc_pairs, anc_pairs_exp)
        test.assert_array_equal(pns_pairs, pns_pairs_exp)

    def test_gen_dataset_two2one_case2_chunked(self):
        '''Tests how gen_dataset_two2one_case2() works chunked.'''

        print("\n")

        dataset = create_test_dataset_huge()

        batch_size = 3
        batch_count = dataset.init_size // batch_size
        end = batch_count * batch_size

        # Get the entire-dataset result.

        init_to_whole, resp_to_whole = dataset.to_gateway
        init_from_whole, resp_from_whole = dataset.from_gateway

        anc_pairs_whole, pns_pairs_whole = common.gen_dataset_two2one_case2(
            init_to_whole,
            init_from_whole,
            resp_to_whole,
            resp_from_whole,
            dataset.delays,
            shift_init=0)

        # Get the batch-sized results.
        for i, start in enumerate(range(0, end, batch_size)):

            stop = start + batch_size
            print(f"Batch {i=}:  {start=} => {stop=}")

            chunk = dataset.chunk(
                0, dataset.init_size, start, stop, True, stop == end)

            init_to_chunk, resp_to_chunk = chunk.to_gateway
            init_from_chunk, resp_from_chunk = chunk.from_gateway

            assert init_to_chunk.shape == (dataset.init_size, 50, 2)

            if stop != end:
                assert resp_to_chunk.shape == ((batch_size + 1), 50, 2)
            else:
                assert resp_to_chunk.shape == (batch_size, 50, 2)

            assert init_from_chunk.shape == (dataset.init_size, 50, 2)

            if stop != end:
                assert resp_from_chunk.shape == ((batch_size + 1), 50, 2)
            else:
                assert resp_from_chunk.shape == (batch_size, 50, 2)

            anc_pairs_chunk, pns_pairs_chunk = common.gen_dataset_two2one_case2(
                init_to_chunk,
                init_from_chunk,
                resp_to_chunk,
                resp_from_chunk,
                chunk.delays,
                shift_init=start)

            if stop != end:
                test.assert_array_equal(
                    anc_pairs_chunk,
                    anc_pairs_whole[(start * (dataset.init_size - 1)):(stop * (dataset.init_size - 1)), :, :])
                test.assert_array_equal(
                    pns_pairs_chunk,
                    pns_pairs_whole[(start * (dataset.init_size - 1)):(stop * (dataset.init_size - 1)), :, :])
            else:
                test.assert_array_equal(
                    anc_pairs_chunk,
                    anc_pairs_whole[(start * (dataset.init_size - 1)):((stop - 1) * (dataset.init_size - 1)), :, :])
                test.assert_array_equal(
                    pns_pairs_chunk,
                    pns_pairs_whole[(start * (dataset.init_size - 1)):((stop - 1) * (dataset.init_size - 1)), :, :])

        print()

    def test_gen_dataset_two2one_case2_chunked_manual(self):
        '''Tests how gen_dataset_two2one_case2() works chunked, manually.'''

        print("\n")

        batch_size = 2
        dataset = create_test_dataset_medium()
        num_init = dataset.init_size

        anc_pairs_exp = []
        pns_pairs_exp = []

        anc_pairs_exp.append(np.array([
            [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [1, 1], [1, 1]],
            [[0, 0], [0, 0], [1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]],
            [[0, 0], [0, 0], [1, 1], [1, 1], [3, 3], [3, 3], [4, 4], [4, 4]],
            [[0, 0], [0, 0], [1, 1], [1, 1], [4, 4], [4, 4], [5, 5], [5, 5]],
            [[0, 0], [0, 0], [1, 1], [1, 1], [2, 2], [2, 2], [5, 5], [5, 5]],
            [[0, 0], [0, 0], [1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]],
            [[1, 1], [1, 1], [1, 1], [1, 1], [2, 2], [2, 2], [2, 2], [2, 2]],
            [[1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3], [4, 4], [4, 4]],
            [[1, 1], [1, 1], [2, 2], [2, 2], [4, 4], [4, 4], [5, 5], [5, 5]],
            [[0, 0], [0, 0], [1, 1], [1, 1], [2, 2], [2, 2], [5, 5], [5, 5]],
        ]))

        anc_pairs_exp.append(np.array([
            [[0, 0], [0, 0], [1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]],
            [[1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3], [4, 4], [4, 4]],
            [[2, 2], [2, 2], [2, 2], [2, 2], [3, 3], [3, 3], [3, 3], [3, 3]],
            [[2, 2], [2, 2], [3, 3], [3, 3], [4, 4], [4, 4], [5, 5], [5, 5]],
            [[0, 0], [0, 0], [2, 2], [2, 2], [3, 3], [3, 3], [5, 5], [5, 5]],
            [[0, 0], [0, 0], [1, 1], [1, 1], [3, 3], [3, 3], [4, 4], [4, 4]],
            [[1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3], [4, 4], [4, 4]],
            [[2, 2], [2, 2], [3, 3], [3, 3], [4, 4], [4, 4], [5, 5], [5, 5]],
            [[3, 3], [3, 3], [3, 3], [3, 3], [4, 4], [4, 4], [4, 4], [4, 4]],
            [[0, 0], [0, 0], [3, 3], [3, 3], [4, 4], [4, 4], [5, 5], [5, 5]],
        ]))

        anc_pairs_exp.append(np.array([
            [[0, 0], [0, 0], [1, 1], [1, 1], [4, 4], [4, 4], [5, 5], [5, 5]],
            [[1, 1], [1, 1], [2, 2], [2, 2], [4, 4], [4, 4], [5, 5], [5, 5]],
            [[2, 2], [2, 2], [3, 3], [3, 3], [4, 4], [4, 4], [5, 5], [5, 5]],
            [[0, 0], [0, 0], [3, 3], [3, 3], [4, 4], [4, 4], [5, 5], [5, 5]],
            [[4, 4], [4, 4], [4, 4], [4, 4], [5, 5], [5, 5], [5, 5], [5, 5]]
        ]))

        pns_pairs_exp.append(np.array([
            [[10, 10], [10, 10], [10, 10], [10, 10], [11, 11], [11, 11], [11, 11], [11, 11]],
            [[10, 10], [10, 10], [11, 11], [11, 11], [12, 12], [12, 12], [13, 13], [13, 13]],
            [[10, 10], [10, 10], [11, 11], [11, 11], [13, 13], [13, 13], [14, 14], [14, 14]],
            [[10, 10], [10, 10], [11, 11], [11, 11], [14, 14], [14, 14], [15, 15], [15, 15]],
            [[10, 10], [10, 10], [11, 11], [11, 11], [12, 12], [12, 12], [15, 15], [15, 15]],
            [[10, 10], [10, 10], [11, 11], [11, 11], [12, 12], [12, 12], [13, 13], [13, 13]],
            [[11, 11], [11, 11], [11, 11], [11, 11], [12, 12], [12, 12], [12, 12], [12, 12]],
            [[11, 11], [11, 11], [12, 12], [12, 12], [13, 13], [13, 13], [14, 14], [14, 14]],
            [[11, 11], [11, 11], [12, 12], [12, 12], [14, 14], [14, 14], [15, 15], [15, 15]],
            [[10, 10], [10, 10], [11, 11], [11, 11], [12, 12], [12, 12], [15, 15], [15, 15]],
        ]))

        pns_pairs_exp.append(np.array([
            [[10, 10], [10, 10], [11, 11], [11, 11], [12, 12], [12, 12], [13, 13], [13, 13]],
            [[11, 11], [11, 11], [12, 12], [12, 12], [13, 13], [13, 13], [14, 14], [14, 14]],
            [[12, 12], [12, 12], [12, 12], [12, 12], [13, 13], [13, 13], [13, 13], [13, 13]],
            [[12, 12], [12, 12], [13, 13], [13, 13], [14, 14], [14, 14], [15, 15], [15, 15]],
            [[10, 10], [10, 10], [12, 12], [12, 12], [13, 13], [13, 13], [15, 15], [15, 15]],
            [[10, 10], [10, 10], [11, 11], [11, 11], [13, 13], [13, 13], [14, 14], [14, 14]],
            [[11, 11], [11, 11], [12, 12], [12, 12], [13, 13], [13, 13], [14, 14], [14, 14]],
            [[12, 12], [12, 12], [13, 13], [13, 13], [14, 14], [14, 14], [15, 15], [15, 15]],
            [[13, 13], [13, 13], [13, 13], [13, 13], [14, 14], [14, 14], [14, 14], [14, 14]],
            [[10, 10], [10, 10], [13, 13], [13, 13], [14, 14], [14, 14], [15, 15], [15, 15]],
        ]))

        pns_pairs_exp.append(np.array([
            [[10, 10], [10, 10], [11, 11], [11, 11], [14, 14], [14, 14], [15, 15], [15, 15]],
            [[11, 11], [11, 11], [12, 12], [12, 12], [14, 14], [14, 14], [15, 15], [15, 15]],
            [[12, 12], [12, 12], [13, 13], [13, 13], [14, 14], [14, 14], [15, 15], [15, 15]],
            [[10, 10], [10, 10], [13, 13], [13, 13], [14, 14], [14, 14], [15, 15], [15, 15]],
            [[14, 14], [14, 14], [14, 14], [14, 14], [15, 15], [15, 15], [15, 15], [15, 15]]
        ]))

        print(f"{len(anc_pairs_exp)=}\n{anc_pairs_exp=}\n")
        print(f"{len(pns_pairs_exp)=}\n{pns_pairs_exp=}\n")

        for i, start in enumerate(range(0, num_init, batch_size)):

            stop = start + batch_size
            print(f"Batch {i=}:  {start=} => {stop=}")

            chunk = dataset.chunk(
                0, num_init, start, stop, True, stop == num_init)

            init_to_chunk, resp_to_chunk = chunk.to_gateway
            init_from_chunk, resp_from_chunk = chunk.from_gateway

            assert init_to_chunk.shape == (num_init, 2, 2)

            if stop != num_init:
                assert resp_to_chunk.shape == ((batch_size + 1), 2, 2)
            else:
                assert resp_to_chunk.shape == (batch_size, 2, 2)

            assert init_from_chunk.shape == (num_init, 2, 2)

            if stop != num_init:
                assert resp_from_chunk.shape == ((batch_size + 1), 2, 2)
            else:
                assert resp_from_chunk.shape == (batch_size, 2, 2)

            anc_pairs_chunk, pns_pairs_chunk = common.gen_dataset_two2one_case2(
                init_to_chunk,
                init_from_chunk,
                resp_to_chunk,
                resp_from_chunk,
                chunk.delays,
                shift_init=start)

            test.assert_array_equal(
                anc_pairs_chunk,
                anc_pairs_exp[i])
            test.assert_array_equal(
                pns_pairs_chunk,
                pns_pairs_exp[i])

        print()

    def test_pad(self):
        a=np.array([[2, 5, 6, 8, np.nan]])
        b=common.pad(a, 7, np.nan, axis=1)
        exp=np.array([[2, 5, 6, 8, np.nan, np.nan, np.nan]])
        test.assert_array_equal(b, exp)

    def test_stack(self):
        a=np.ones((3, 4, 5))
        b=np.random.randint(2, 9, size=a.shape)
        s=common.stack(a, b)
        exp=np.zeros((3, 4, 5, 2))
        exp[..., 0]=a
        exp[..., 1]=b
        test.assert_array_equal(s, exp)

    def test_unstack(self):
        a=np.ones((3, 4, 5))
        b=np.random.randint(2, 9, size=a.shape)
        s=common.stack(a, b)
        res_a, res_b=common.unstack(s)
        test.assert_array_equal(a, res_a)
        test.assert_array_equal(b, res_b)

    def test_discard_along_axes(self):
        data1=np.array([[[1, 3, 5, 7, 9],
                           [1, 3, 5, 7, 9],
                           [1, 3, 5, 7, np.nan],
                           [np.nan, np.nan, np.nan, np.nan, np.nan]],

                          [[1, 3, 5, 7, 9],
                           [1, 3, 5, 7, 9],
                           [1, 3, 5, 7, 9],
                           [np.nan, np.nan, np.nan, np.nan, np.nan]],

                          [[1, 3, 5, 7, 9],
                           [1, 3, 5, 7, 9],
                           [1, 3, 5, 7, 9],
                           [1, 7, np.nan, np.nan, np.nan]]])
        acks1=np.random.randint(0, 9, size=data1.shape)
        a1=common.stack(data1, acks1)
        self.assertEqual(a1.shape, (3, 4, 5, 2))

        data2=np.array([[[2, 4, 8, 5, 2],
                           [1, 3, 1, 6, 9],
                           [1, 8, 5, 7, 8],
                           [np.nan, np.nan, np.nan, np.nan, np.nan]],

                          [[1, 3, 5, 3, 9],
                           [3, 9, 9, 2, 8],
                           [1, 3, 5, 1, 9],
                           [np.nan, np.nan, np.nan, np.nan, np.nan]],

                          [[0, 3, 5, 7, 9],
                           [5, 4, 8, 7, 9],
                           [6, 3, 8, 7, 3],
                           [7, 7, np.nan, np.nan, np.nan]]])
        acks2=np.random.randint(0, 9, size=data2.shape)
        a2=common.stack(data2, acks2)
        self.assertEqual(a2.shape, (3, 4, 5, 2))

        res_a1, res_a2=common.discard_along_axes(a1, a2)
        exp1=np.array([[[1, 3, 5, 7, 9],
                          [1, 3, 5, 7, 9]],

                         [[1, 3, 5, 7, 9],
                          [1, 3, 5, 7, 9]],

                         [[1, 3, 5, 7, 9],
                          [1, 3, 5, 7, 9]]])
        test.assert_array_equal(res_a1[..., 0], exp1)
        exp2=np.array([[[2, 4, 8, 5, 2],
                          [1, 3, 1, 6, 9]],

                         [[1, 3, 5, 3, 9],
                          [3, 9, 9, 2, 8]],

                         [[0, 3, 5, 7, 9],
                          [5, 4, 8, 7, 9]]])
        test.assert_array_equal(res_a2[..., 0], exp2)


if __name__ == '__main__':
    unittest.main()
