import cv2
import numpy as np
import pandas as pd
# from tensorflow import keras
import keras
from sklearn.model_selection import train_test_split
from os.path import join

DATA_PATH = './data'
BATCH = 8


def generator(x_data, y_data, batch_size=BATCH, augment=False):
    while True:
        x_batch, y_batch = [], []
        for x, y in zip(x_data, y_data):
            if y < 0.4 and np.random.random() > y * 2:
                continue
            path = join(DATA_PATH, x)
            x_batch.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
            y_batch.append(y)
            if len(x_batch) >= batch_size:
                yield np.array(x_batch), np.array(y_batch)
                x_batch, y_batch = [], []


df = pd.read_csv('data/driving_log.csv')
x = df.center.values
y = df.steering.values

xt, xv, yt, yv = train_test_split(x, y)

model = keras.models.Sequential()
model.add(keras.layers.Cropping2D(
    cropping=((60, 20), (0, 0)),
    input_shape=(160, 320, 3)))
model.add(keras.layers.Lambda(lambda x: (x / 255.0) - 0.5))
model.add(keras.layers.Conv2D(24, 5, activation='relu'))
model.add(keras.layers.MaxPool2D((2, 2), (2, 2)))
model.add(keras.layers.Conv2D(36, 5, activation='relu'))
model.add(keras.layers.MaxPool2D((2, 2), (2, 2)))
model.add(keras.layers.Conv2D(48, 5, activation='relu'))
model.add(keras.layers.MaxPool2D((2, 2), (2, 2)))
model.add(keras.layers.Conv2D(64, 5, activation='relu'))
model.add(keras.layers.MaxPool2D((2, 2), (2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(1))
print(model.summary())

model.compile(optimizer='adam', loss='mse')
model.fit_generator(
    generator=generator(xt, yt),
    steps_per_epoch=yt.size // BATCH,
    epochs=50,
    callbacks=[keras.callbacks.ModelCheckpoint(
        './model.h5',
        save_best_only=True)],
    validation_data=generator(xv, yv),
    validation_steps=yv.size // BATCH
)
