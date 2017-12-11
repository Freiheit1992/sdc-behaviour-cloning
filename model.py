import cv2
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from os.path import join

from keras.utils import plot_model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Cropping2D, Lambda, Conv2D, MaxPool2D,\
    Flatten, Dropout, Dense, concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping

DATA_PATH = './data'
BATCH = 8


def generator(images, speed, throttle, angles, batch_size=BATCH, training=False):
    while True:
        x_batch, s_batch, a_batch, t_batch = [], [], [], []
        for x, s, t, a in zip(images, speed, throttle, angles):
            aa = np.abs(a)
            if training and np.random.random() > np.min([.1, aa * 5]):
                # Subsample small angles
                continue
            path = join(DATA_PATH, x)
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            if training and np.random.random() > 0.5:
                image = np.fliplr(image)
                a = -a
            x_batch.append(image)
            s_batch.append(s)
            a_batch.append(a)
            t_batch.append(t)
            if len(x_batch) >= batch_size:
                yield [np.array(x_batch), np.array(s_batch)], [np.array(a_batch), np.array(t_batch)]
                x_batch, s_batch, a_batch, t_batch = [], [], [], []


df = pd.read_csv('data/driving_log.csv')
center = df.center.values
speed = df.speed.values
throttle = df.throttle.values
y = df.steering.values

ct, cv, st, sv, tt, tv, yt, yv = train_test_split(center, speed, throttle, y, test_size=0.2)


# Convolution tower
image = Input(shape=(160, 320, 3))
x = Cropping2D(cropping=((60, 20), (0, 0)))(image)
x = Lambda(lambda x: (x / 255.0) - 0.5)(x)
x = Conv2D(filters=24, kernel_size=5, activation='relu')(x)
x = MaxPool2D(strides=(2, 2))(x)
x = Conv2D(filters=36, kernel_size=5, activation='relu')(x)
x = MaxPool2D(strides=(2, 2))(x)
x = Conv2D(filters=48, kernel_size=5, activation='relu')(x)
x = MaxPool2D(strides=(2, 2))(x)
x = Conv2D(filters=64, kernel_size=3, activation='relu')(x)
x = Conv2D(filters=64, kernel_size=3, activation='relu')(x)
x = Flatten()(x)

# Concatenate image data and velocity
velocity = Input(shape=(1, ))
x = concatenate([x, velocity])
x = Dense(200, activation='relu', kernel_regularizer=l2())(x)

# Angle tower
xa = Dense(100, activation='relu', kernel_regularizer=l2())(x)
xa = Dense(50, activation='relu', kernel_regularizer=l2())(xa)
xa = Dense(10, activation='relu', kernel_regularizer=l2())(xa)
xa = Dense(1)(xa)

# Throttle tower
xt = Dropout(0.5)(x)
xt = Dense(100)(xt)
xt = Dropout(0.5)(xt)
xt = Dense(10)(xt)
xt = Dropout(0.5)(xt)
xt = Dense(1)(xt)

# Define model with multiple inputs and outputs
model = Model(inputs=[image, velocity], outputs=[xa, xt])
model.compile(optimizer=Adam(lr=5e-4), loss='mse', loss_weights=[1., 0.01])
plot_model(model, 'model.png', show_shapes=True)
print(model.summary())

# Training
callbacks = [
    ModelCheckpoint('./model.h5', save_best_only=True),
    EarlyStopping(patience=3)
]

history = model.fit_generator(
    generator=generator(ct, st, tt, yt, training=True),
    steps_per_epoch=yt.size // BATCH,
    epochs=50,
    callbacks=callbacks,
    validation_data=generator(cv, sv, tv, yv),
    validation_steps=yv.size // BATCH
)

with open('history.p', 'wb') as f:
    pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)
