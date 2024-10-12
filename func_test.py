import unittest
import numpy as np
import torch
from unittest.mock import MagicMock
from sklearn.cluster import KMeans
from funcs import predict_kmeans, predict_cnn, tile_raster

class TestPredictKmeans(unittest.TestCase):

    def test_predict_kmeans(self):
        test_cases = [
            ((3, 64, 64), 4, 16, (1, 64, 64)),
            ((3, 128, 128), 8, 16, (1, 128, 128)),
            ((3, 256, 256), 16, 16, (1, 256, 256)),
            ((3, 256, 128), 8, 16, (1, 256, 128)),
            ((3, 128, 256), 8, 16, (1, 128, 256))
        ]
        for array_shape, num_segments, resolution, expected_shape in test_cases:
            with self.subTest(array_shape=array_shape):
                array = np.random.rand(*array_shape)
                result = predict_kmeans(array, num_segments=num_segments, resolution=resolution)
                self.assertEqual(result.shape, expected_shape)

class TestPredictCnn(unittest.TestCase):

    def test_predict_cnn(self):
        test_cases = [
            ((3, 512, 512), 4, 256, (1, 512, 512), torch.randn(1, 384, 126, 126)),
            ((3, 256, 256), 4, 128, (1, 256, 256), torch.randn(1, 384, 126, 126)),
            ((3, 128, 128), 4, 64, (1, 128, 128), torch.randn(1, 384, 126, 126)),
            ((3, 256, 128), 4, 64, (1, 256, 128), torch.randn(1, 384, 126, 126)),
            ((3, 128, 256), 4, 64, (1, 128, 256), torch.randn(1, 384, 126, 126))
        ]
        for array_shape, num_segments, tile_size, expected_shape, mock_output in test_cases:
            with self.subTest(array_shape=array_shape):
                cnn_model = MagicMock()
                cnn_model.forward = MagicMock(return_value=[None, mock_output])
                array = np.random.rand(*array_shape)
                result = predict_cnn(cnn_model, array, num_segments=num_segments, tile_size=tile_size, device="cpu")
                self.assertEqual(result.shape, expected_shape)

class TestTileRaster(unittest.TestCase):

    def test_tile_raster(self):
        test_cases = [
            ((3, 512, 512), 256, 0, 0, (4, 3, 256, 256)),
            ((3, 500, 500), 256, 12, 12, (4, 3, 256, 256)),
            ((3, 256, 256), 128, 0, 0, (4, 3, 128, 128)),
            ((3, 300, 300), 128, 84, 84, (9, 3, 128, 128)),
            ((3, 256, 128), 128, 0, 0, (2, 3, 128, 128))
        ]
        for array_shape, tile_size, expected_height_pad, expected_width_pad, expected_shape in test_cases:
            with self.subTest(array_shape=array_shape):
                array = np.random.rand(*array_shape)
                tiles, (height_pad, width_pad) = tile_raster(array, tile_size=tile_size)
                self.assertEqual(height_pad, expected_height_pad)
                self.assertEqual(width_pad, expected_width_pad)
                self.assertEqual(tiles.shape, expected_shape)

if __name__ == "__main__":
    unittest.main()
