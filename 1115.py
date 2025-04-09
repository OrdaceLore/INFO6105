import os
import zipfile
import pandas as pd

# ✅ Step 1: 设置 Kaggle 凭据（推荐先手动放 kaggle.json 到 ~/.kaggle/）
os.environ['KAGGLE_USERNAME'] = 'ordacelore'
os.environ['KAGGLE_KEY'] = '6c7cecf1731724d63431f5763d0ef91c'

# ✅ Step 2: 下载 Kaggle 数据集（例子是 Salary Prediction 数据）
dataset = 'amirmahdiabbootalebi/salary-by-job-title-and-country'
os.system(f'kaggle datasets download -d {dataset}')

# ✅ Step 3: 解压文件
zip_file = dataset.split('/')[1] + '.zip'
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall('.')

# ✅ Step 4: 加载 CSV 到 DataFrame
df = pd.read_csv('Salary.csv')
print(df.head())

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Assuming 'Salary' is the target variable
data = df[['Salary']] 

# Scale the data
scaler = MinMaxScaler()
data['Salary'] = scaler.fit_transform(data['Salary'].values.reshape(-1, 1))

# Create sequences for RNN input
def create_sequences(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 10 # Number of previous data points to consider
X, Y = create_sequences(data[['Salary']].values, look_back)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=100, batch_size=32)


predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions) # Inverse transform to get actual salary values

# Print the predictions
print(predictions)