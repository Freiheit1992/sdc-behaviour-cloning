from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, concatenate
from keras.applications.vgg16 import VGG16
from nvidia import preprocess


def get_vgg16(image, velocity):
    # Inputs
    image = preprocess(image)

    # Load pretrained VGG16
    base_model = VGG16(input_tensor=image, weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = concatenate([x, velocity])

    # Fully connected trainable layers
    x = Dense(100, activation='relu')(x)
    x = Dropout(.5)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(.5)(x)
    x = Dense(10, activation='relu')(x)
    x = Dropout(.5)(x)

    # Independent outputs to apply loss weights
    xa = Dense(1, name='steering')(x)
    xt = Dense(1, name='throttle')(x)

    # Define model with multiple inputs and outputs
    model = Model(inputs=[base_model.input, velocity], outputs=[xa, xt])
    return model
