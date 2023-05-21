import tensorflow as tf
from tensorflow.keras.models import load_model
from dataset import gisdata

model = load_model("gis_seg")

data = gisdata()
img = data.full_map()

result = model.fit([img])[0]

cv2.imshow("input", img)
cv2.imshow("output", result)
cv2.waitKey(10000)
