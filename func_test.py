import unittest
import numpy as np
import torch
from unittest.mock import MagicMock
from sklearn.cluster import KMeans
from funcs import predict_kmeans, predict_cnn, tile_raster

class TestPredictKmeans(unittest.TestCase):

    def test_predict_kmeans_shape(self):
        array = np.random.rand(3, 64, 64)  # 3-channel, 64x64 array
        result = predict_kmeans(array, num_segments=4, resolution=16)
        self.assertEqual(result.shape, (1, 64, 64))

class TestPredictCnn(unittest.TestCase):

    def test_predict_cnn_shape(self):
        cnn_model = MagicMock()
        cnn_model.forward = MagicMock(return_value=[None, torch.randn(1, 3, 256, 256)])
        array = np.random.rand(3, 512, 512)
        result = predict_cnn(cnn_model, array, num_segments=4, tile_size=256, device="cpu")
        self.assertEqual(result.shape, (1, 512, 512))

class TestTileRaster(unittest.TestCase):

    def test_tile_raster_output_shape(self):
        array = np.random.rand(3, 512, 512)
        tiles, (height_pad, width_pad) = tile_raster(array, tile_size=256)
        self.assertEqual(tiles.shape, (4, 3, 256, 256))  # 4 tiles for 512x512 with no padding

    def test_tile_raster_with_padding(self):
        array = np.random.rand(3, 500, 500)
        tiles, (height_pad, width_pad) = tile_raster(array, tile_size=256)
        self.assertEqual(height_pad, 12)  # 512 - 500 = 12 padding
        self.assertEqual(width_pad, 12)
        self.assertEqual(tiles.shape, (4, 3, 256, 256))  # 4 tiles after padding

if __name__ == "__main__":
    unittest.main()
