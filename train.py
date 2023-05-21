from model import autoencoder
from dataset import gisdata
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import keras
import cv2
import numpy as np

# Generators
print("Getting dataset")
generator = ImageDataGenerator()
data = generator.flow_from_directory(
        "./local_data",
        class_mode="input",
        classes=["maps"],
        batch_size=8,
        target_size=(512,512),
    )

# Model
print("Getting model")
model = autoencoder([512,512,3])
model.compile(optimizer='adam', loss='mse')
model.summary()

# Train
print("Training!")
model.fit(data,
          epochs=10,
          workers=4,
          use_multiprocessing=True,)

model.save("gis_seg")
