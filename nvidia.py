from keras.regularizers import l2
from keras.models import Model
from keras.layers import Cropping2D, Lambda, Conv2D, MaxPool2D, \
    Flatten, Dropout, Dense, concatenate


def preprocess(image):
    x = Cropping2D(cropping=((60, 20), (0, 0)))(image)
    x = Lambda(lambda x: (x / 127.5) - 1)(x)
    return x


def get_nvidia(image, velocity):
    # Inputs
    x = preprocess(image)

    # Convolution tower
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
    x = concatenate([x, velocity])
    x = Dense(200, activation='relu', kernel_regularizer=l2())(x)

    # Steering tower
    xa = Dense(100, activation='relu', kernel_regularizer=l2())(x)
    xa = Dense(50, activation='relu')(xa)
    xa = Dense(10, activation='relu')(xa)
    xa = Dense(1, name='steering')(xa)

    # Throttle tower
    xt = Dense(100, activation='relu')(x)
    xt = Dropout(.5)(xt)
    xt = Dense(10, activation='relu')(xt)
    xt = Dense(1, name='throttle')(xt)

    # Define model with multiple inputs and outputs
    model = Model(inputs=[image, velocity], outputs=[xa, xt])
    return model
