import numpy as np
import torch
from sklearn.cluster import KMeans
import numpy as np

# Predict coverage map using kmeans
def predict_kmeans(array, num_segments=16, resolution=16):
    # Instantiate kmeans model
    kmeans = KMeans(n_clusters=num_segments)

    # Pad to resolution
    channel_pad = (0, 0)
    height_pad = (0, resolution - array.shape[1] % resolution)
    width_pad = (0, resolution - array.shape[2] % resolution)
    array_padded = np.pad(
        array, (channel_pad, height_pad, width_pad), mode="constant"
    )

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

    # Fit kmeans model to random subset
    size = 10000 if array_2d.shape[0] > 10000 else array_2d.shape[0]
    idx = np.random.randint(0, array_2d.shape[0], size=size)
    kmeans = kmeans.fit(array_2d[idx])

    # Get clusters
    clusters = kmeans.predict(array_2d)

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

    # Convert from tiles (Ht*Wt, C, tile_size, tile_size) to (C, H, W)
    _, C, _, _ = coverage_map.shape
    
    # Calculate full height (H) and width (W)
    Ht = (array.shape[1] + height_pad) // tile_size
    Wt = (array.shape[2] + width_pad) // tile_size
    H = Ht * tile_size
    W = Wt * tile_size
    
    # Reshape the tiled array into the full form
    coverage_map = coverage_map.reshape(Ht, Wt, C, tile_size, tile_size)
    
    # Transpose to move tiles into the correct position
    coverage_map = coverage_map.transpose(2, 0, 3, 1, 4)
    
    # Reshape into the full (C, H, W) array
    coverage_map = coverage_map.reshape(C, H, W)

    # Perform kmeans to get num_segments clusters
    coverage_map = predict_kmeans(
        coverage_map,
        num_segments=num_segments,
        resolution=1,
    )

    # Get rid of padding
    coverage_map = coverage_map[:, : array.shape[1], : array.shape[2]]

    return coverage_map

# Tile raster for CNN or K-means
def tile_raster(array, tile_size):
    # Pad to tile_size
    padding = lambda shape: 0 if shape % tile_size == 0 else tile_size - shape % tile_size
    channel_pad = (0, 0)
    height_pad = (0, padding(array.shape[1]))
    width_pad = (0, padding(array.shape[2]))
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
