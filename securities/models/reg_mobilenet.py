from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.applications.mobilenet import MobileNet

def reg_mobilenet(input_shape):

    base_model = MobileNet(input_shape=input_shape, include_top=False, weights=None)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.2))
    top_model.add(Dense(1, activation='linear'))
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])
    return model
