import pandas as pd
import time, datetime
from math import sqrt
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
from keras.layers import Input, Dense, LSTM
from keras.models import Model
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.models import load_model


df_data_5minute=pd.read_csv('raw_data.csv')
df_data_5minute.drop('Unnamed: 0', axis=1, inplace=True)

df=df_data_5minute
close = df['close']
df.drop(labels=['close'], axis=1,inplace = True)
df.insert(0, 'close', close)

data_train =df.iloc[:30000, :]
data_test = df.iloc[30000:, :]
print(data_train.shape, data_test.shape)


scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_train)

data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)


output_dim = 1
batch_size = 1000
epochs = 500
seq_len = 5
hidden_size = 128


TIME_STEPS = 5
INPUT_DIM = 14

lstm_units = 64 
X_train = np.array([data_train[i : i + seq_len, :] for i in range(data_train.shape[0] - seq_len)])
y_train = np.array([data_train[i + seq_len, 0] for i in range(data_train.shape[0]- seq_len)])
X_test = np.array([data_test[i : i + seq_len, :] for i in range(data_test.shape[0]- seq_len)])
y_test = np.array([data_test[i + seq_len, 0] for i in range(data_test.shape[0] - seq_len)])
maxy=y_test.max();
miny=y_test.min();
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

inputs = Input(shape=(TIME_STEPS, INPUT_DIM))

x = Conv1D(filters = 32, kernel_size = 1, activation = 'relu')(inputs)  #, padding = 'same'
x = MaxPooling1D(pool_size = 5)(x)
x = Dropout(0.1)(x)

lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)

output = Dense(1, activation='sigmoid')(lstm_out)

model = Model(inputs=inputs, outputs=output)

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=False)
model.save('model.h5')
#model=load_model('model.h5')


y_pred = model.predict(X_test)



data_train=scaler.inverse_transform(data_train);
data_test=scaler.inverse_transform(data_test);
y_test = np.array([data_test[i + seq_len, 0] for i in range(data_test.shape[0] - seq_len)])
y_train = np.array([data_train[i + seq_len, 0] for i in range(data_train.shape[0]- seq_len)])
y_raw=np.hstack((y_train,y_test))

#RMSE
print('MSE Train loss:', model.evaluate(X_train, y_train, batch_size=batch_size))
print('MSE Test loss:', model.evaluate(X_test, y_test, batch_size=batch_size))
Rmse = sqrt(mean_squared_error(y_test, y_pred))
print('RMSE: ', Rmse)


plt.plot(np.arange(len(y_raw)), np.hstack((y_train,y_test)) , 'b', label="Raw Data")
plt.plot(np.arange(len(y_train),len(y_raw)),y_pred,'r', label="Prediction")
plt.legend()
plt.show()


