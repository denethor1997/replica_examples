from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout
from keras.layers.recurrent import LSTM

def reg_lstm(input_dimension):
    model = Sequential()
    model.add(LSTM(input_dim=input_dimension, output_dim=input_dimension*2+1, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim=input_dimension*2+1, output_dim=input_dimension*2+1, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(input_dim=input_dimension*2+1, output_dim=1))
    model.add(Activation('linear'))
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])
    return model
