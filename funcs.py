import numpy as np
import torch
from sklearn.cluster import KMeans
import numpy as np
from osgeo import gdal
import os
from tempfile import gettempdir
from qgis.core import QgsRasterLayer, QgsProject, QgsMessageLog, Qgis

# Predict coverage map using kmeans
def predict_kmeans(array, num_segments=16, resolution=16):
    QgsMessageLog.logMessage("Starting kmeans", "Segmenter", Qgis.info)
    QgsMessageLog.logMessage(f"Array shape: {array.shape}", "Segmenter", Qgis.info)

    # Instantiate kmeans model
    kmeans = KMeans(n_clusters=num_segments)

    # Pad to resolution
    channel_pad = (0, 0)
    height_pad = (0, array.shape[1] % resolution)
    width_pad = (0, array.shape[2] % resolution)
    array_padded = np.pad(
        array, (channel_pad, height_pad, width_pad), mode="constant"
    )
    QgsMessageLog.logMessage(f"Padded array shape: {array_padded.shape}", "Segmenter", Qgis.info)

    # Reshape into 2d
    array_2d = array_padded.reshape(
        array_padded.shape[0],
        array_padded.shape[1] // resolution,
        resolution,
        array_padded.shape[2] // resolution,
        resolution,
    )
    array_2d = array_2d.transpose(1, 3, 0, 2, 4)
    array_2d = array_2d.reshape(
        array_2d.shape[0] * array_2d.shape[1],
        array_2d.shape[2] * resolution * resolution,
    )
    QgsMessageLog.logMessage(f"2D array shape: {array_2d.shape}", "Segmenter", Qgis.info)

    # Fit kmeans model to random subset
    size = 10000 if array_2d.shape[0] > 10000 else array_2d.shape[0]
    idx = np.random.randint(0, array_2d.shape[0], size=size)
    kmeans = kmeans.fit(array_2d[idx])

    # Get clusters
    clusters = kmeans.predict(array_2d)
    QgsMessageLog.logMessage(f"Clusters shape: {clusters.shape}", "Segmenter", Qgis.info)

    # Reshape clusters to match map
    clusters = clusters.reshape(
        1,
        1,
        array_padded.shape[1] // resolution,
        array_padded.shape[2] // resolution,
    )

    # Get rid of padding
    clusters = clusters[
        :, :, : array.shape[1] // resolution, : array.shape[2] // resolution
    ]
    QgsMessageLog.logMessage(f"Clusters shape: {clusters.shape}", "Segmenter", Qgis.info)

    # Upsample to original size
    clusters = torch.tensor(clusters)
    clusters = torch.nn.Upsample(
        size=(array.shape[-2], array.shape[-1]), mode="nearest"
    )(clusters.byte())
    clusters = clusters[0]

    return clusters.cpu().numpy()

# Predict coverage map using cnn
def predict_cnn(cnn_model, array, num_segments, tile_size=256, device="cpu"):

    assert array.shape[0] == 3, f"Invalid array shape! \n{array.shape}"

    # Tile raster
    tiles, (height_pad, width_pad) = tile_raster(array, tile_size)

    # Convert to float range [0, 1]
    tiles = tiles.astype("float32") / 255

    # Predict vectors
    batch_size = 1
    coverage_map = []
    for i in range(0, tiles.shape[0], batch_size):
        with torch.no_grad():
            tile = torch.tensor(tiles[i : i + batch_size]).to(device)
            result = cnn_model.forward(tile)
        vectors = result[1].cpu().numpy()
        coverage_map.append(vectors)
    coverage_map = np.concatenate(coverage_map, axis=0)

    # Convert from tiles to one big map
    coverage_map = coverage_map.reshape(
        (array.shape[1] + height_pad) // tile_size,
        (array.shape[2] + width_pad) // tile_size,
        coverage_map.shape[1],
        coverage_map.shape[2],
        coverage_map.shape[3],
    )
    coverage_map = coverage_map.transpose(2, 0, 3, 1, 4)
    coverage_map = coverage_map.reshape(
        coverage_map.shape[0],
        coverage_map.shape[1] * coverage_map.shape[2],
        coverage_map.shape[3] * coverage_map.shape[4],
    )

    # Perform kmeans to get num_segments clusters
    coverage_map = predict_kmeans(
        coverage_map,
        num_segments=num_segments,
        resolution=1,
    )

    # Upsample
    coverage_map = torch.tensor(coverage_map)
    coverage_map = torch.unsqueeze(coverage_map, dim=0)
    coverage_map = torch.nn.Upsample(
        size=(array.shape[1]+height_pad, array.shape[2]+width_pad), mode="nearest"
    )(coverage_map)

    # Get rid of padding
    coverage_map = coverage_map[0, :, : array.shape[1], : array.shape[2]]

    return coverage_map.cpu().numpy()

# Tile raster for CNN or K-means
def tile_raster(array, tile_size):
    # Pad to tile_size
    channel_pad = (0, 0)
    height_pad = (0, tile_size - array.shape[1] % tile_size)
    width_pad = (0, tile_size - array.shape[2] % tile_size)
    array_padded = np.pad(
        array,
        (channel_pad, height_pad, width_pad),
        mode="constant",
    )

    # Reshape into tiles
    tiles = array_padded.reshape(
        array_padded.shape[0],
        array_padded.shape[1] // tile_size,
        tile_size,
        array_padded.shape[2] // tile_size,
        tile_size,
    )
    tiles = tiles.transpose(1, 3, 0, 2, 4)
    tiles = tiles.reshape(
        tiles.shape[0] * tiles.shape[1],
        array_padded.shape[0],
        tile_size,
        tile_size,
    )

    return tiles, (height_pad[1], width_pad[1])

# Render raster from array
def render_raster(array, bounding_box, layer_name, epsg):
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        os.path.join(gettempdir(), layer_name + ".tif"),
        array.shape[2],
        array.shape[1],
        array.shape[0],
        gdal.GDT_Byte,
    )

    out_srs = gdal.osr.SpatialReference()
    out_srs.ImportFromEPSG(epsg)

    dataset.SetProjection(out_srs.ExportToWkt())

    dataset.SetGeoTransform(
        (
            bounding_box.xMinimum(),  # 0
            bounding_box.width() / array.shape[2],  # 1
            0,  # 2
            bounding_box.yMaximum(),  # 3
            0,  # 4
            -bounding_box.height() / array.shape[1],
        )
    )

    for c in range(array.shape[0]):
        dataset.GetRasterBand(c + 1).WriteArray(array[c, :, :])
    dataset = None

    raster_layer = QgsRasterLayer(
        os.path.join(gettempdir(), layer_name + ".tif"), layer_name
    )
    raster_layer.renderer().setOpacity(1.0)

    QgsProject.instance().addMapLayer(raster_layer, True)
