import glob
import cv2
from multiprocessing import Pool


def convert(og, jpg):
    img = cv2.imread(og)
    cv2.imwrite(jpg, img)

if __name__ == "__main__":
    og = glob.glob("data/maps/*.jp2")
    jpg = ["/".join(f.split(".")[:-1])+".jpg" for f in og]

    pool = Pool(16)
    pool.starmap(convert, zip(og, jpg))