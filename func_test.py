import unittest
import numpy as np
import torch
from unittest.mock import MagicMock
from sklearn.cluster import KMeans
from funcs import predict_kmeans, predict_cnn, tile_raster

class TestPredictKmeans(unittest.TestCase):

    def test_predict_kmeans_shape_64x64(self):
        array = np.random.rand(3, 64, 64)  # 3-channel, 64x64 array
        result = predict_kmeans(array, num_segments=4, resolution=16)
        self.assertEqual(result.shape, (1, 64, 64))

    def test_predict_kmeans_shape_128x128(self):
        array = np.random.rand(3, 128, 128)  # 3-channel, 128x128 array
        result = predict_kmeans(array, num_segments=8, resolution=16)
        self.assertEqual(result.shape, (1, 128, 128))

    def test_predict_kmeans_shape_256x256(self):
        array = np.random.rand(3, 256, 256)  # 3-channel, 256x256 array
        result = predict_kmeans(array, num_segments=16, resolution=16)
        self.assertEqual(result.shape, (1, 256, 256))

    def test_predict_kmeans_shape_256x128(self):
        array = np.random.rand(3, 256, 128)
        result = predict_kmeans(array, num_segments=8, resolution=16)
        self.assertEqual(result.shape, (1, 256, 128))

    def test_predict_kmeans_shape_128x256(self):
        array = np.random.rand(3, 128, 256)
        result = predict_kmeans(array, num_segments=8, resolution=16)
        self.assertEqual(result.shape, (1, 128, 256))

class TestPredictCnn(unittest.TestCase):

    def test_predict_cnn_shape_512x512(self):
        cnn_model = MagicMock()
        cnn_model.forward = MagicMock(return_value=[None, torch.randn(1, 3, 256, 256)])
        array = np.random.rand(3, 512, 512)
        result = predict_cnn(cnn_model, array, num_segments=4, tile_size=256, device="cpu")
        self.assertEqual(result.shape, (1, 512, 512))

    def test_predict_cnn_shape_256x256(self):
        cnn_model = MagicMock()
        cnn_model.forward = MagicMock(return_value=[None, torch.randn(1, 3, 128, 128)])
        array = np.random.rand(3, 256, 256)
        result = predict_cnn(cnn_model, array, num_segments=4, tile_size=128, device="cpu")
        self.assertEqual(result.shape, (1, 256, 256))

    def test_predict_cnn_shape_128x128(self):
        cnn_model = MagicMock()
        cnn_model.forward = MagicMock(return_value=[None, torch.randn(1, 3, 64, 64)])
        array = np.random.rand(3, 128, 128)
        result = predict_cnn(cnn_model, array, num_segments=4, tile_size=64, device="cpu")
        self.assertEqual(result.shape, (1, 128, 128))

    def test_predict_cnn_shape_256x128(self):
        cnn_model = MagicMock()
        cnn_model.forward = MagicMock(return_value=[None, torch.randn(1, 3, 64, 64)])
        array = np.random.rand(3, 256, 128)
        result = predict_cnn(cnn_model, array, num_segments=4, tile_size=64, device="cpu")
        self.assertEqual(result.shape, (1, 256, 128))

    def test_predict_cnn_shape_128x256(self):
        cnn_model = MagicMock()
        cnn_model.forward = MagicMock(return_value=[None, torch.randn(1, 3, 64, 64)])
        array = np.random.rand(3, 128, 256)
        result = predict_cnn(cnn_model, array, num_segments=4, tile_size=64, device="cpu")
        self.assertEqual(result.shape, (1, 128, 256))

class TestTileRaster(unittest.TestCase):

    def test_tile_raster_output_shape_512x512(self):
        array = np.random.rand(3, 512, 512)
        tiles, (height_pad, width_pad) = tile_raster(array, tile_size=256)
        self.assertEqual(tiles.shape, (4, 3, 256, 256))  # 4 tiles for 512x512 with no padding

    def test_tile_raster_with_padding_500x500(self):
        array = np.random.rand(3, 500, 500)
        tiles, (height_pad, width_pad) = tile_raster(array, tile_size=256)
        self.assertEqual(height_pad, 12)  # 512 - 500 = 12 padding
        self.assertEqual(width_pad, 12)
        self.assertEqual(tiles.shape, (4, 3, 256, 256))  # 4 tiles after padding

    def test_tile_raster_output_shape_256x256(self):
        array = np.random.rand(3, 256, 256)
        tiles, (height_pad, width_pad) = tile_raster(array, tile_size=128)
        self.assertEqual(tiles.shape, (4, 3, 128, 128))  # 4 tiles for 256x256 with no padding

    def test_tile_raster_with_padding_300x300(self):
        array = np.random.rand(3, 300, 300)
        tiles, (height_pad, width_pad) = tile_raster(array, tile_size=128)
        self.assertEqual(height_pad, 84)  # 128 * 3 = 384, so padding = 384 - 300
        self.assertEqual(width_pad, 84)
        self.assertEqual(tiles.shape, (9, 3, 128, 128))  # 9 tiles after padding

    def test_tile_raster_output_shape_256x128(self):
        array = np.random.rand(3, 256, 128)
        tiles, (height_pad, width_pad) = tile_raster(array, tile_size=128)
        self.assertEqual(tiles.shape, (2, 3, 128, 128))

if __name__ == "__main__":
    unittest.main()
