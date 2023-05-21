import cv2
import numpy as np
import glob
import random
from tqdm import tqdm

from multiprocessing import Pool

class gisdata():
    def __init__(self, data_path="./data/maps"):

        self.map_images = glob.glob("./data/maps" + "/*.jp2")
        self.n = len(self.map_images)
        self.n_channels = 3
        self.dim = [512,512]
        self.map_batch = 8
        self.tiles_per_map = 128//self.map_batch
        self.threads = Pool(self.map_batch)

    def full_map(self, map_id=None):
        # Select and read random map
        if map_id is None:
            rand_map = random.choice(self.map_images)
        else:
            rand_map = map_id
        map_bitmap = cv2.imread(rand_map)
        return map_bitmap

    def tile(self, size=(512,512), map_id=None):

        # Select and read random map
        map_bitmap = self.full_map(map_id=map_id)

        x_rand = random.randint(0, map_bitmap.shape[0]-size[0])
        y_rand = random.randint(0, map_bitmap.shape[1]-size[1])

        map_tile = map_bitmap[x_rand:x_rand+size[0], y_rand:y_rand+size[1], :]

        return map_tile

    def tiles(self, size=(512,512), map_id=None, count=1):
        map_tiles = []
        for t in range(count):
            map_tiles.append(self.tile(size, map_id))
        return map_tiles


if __name__ == "__main__":

    dataset = gisdata()
    tile = dataset.tile()
    cv2.imshow("tile", tile)
    cv2.waitKey(0)
