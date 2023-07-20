# Predicting-Stock-Market-Index-Using-LSTM-yFinance
This project run the program by TSLA stock from 2010-2023

Pls install program below for first step:
1. pip install numpy
2. pip install matplotlib
3. pip install yfinance
4. pip install pandas

# Start with improt program<br>
import pandas as pd <br>
import numpy as np <br>
import matplotlib.pyplot as plt <br>
import yfinance as yf <br>
import plotly.graph_objects as go <br><br>

tsla = yf.Ticker("TSLA") <br>
hist = tsla.history(period="max") <br>
hist["Open"].plot(figsize=(15, 5), color='blue', title="TSLA Stock Price") <br>
plt.show() <br>

![image](https://github.com/Kanangnut/Predicting-Stock-Market-Index-Using-LSTM-yFinance/assets/130201193/aebe4050-be8f-40e8-94ed-701eabbaa637)
 <br>
# Importing the training set<br>
dataset_train = pd.read_csv('../Simulation-HFT-Sentiment-Analyze-From-yFinance-byCat/TSLA.csv') <br>
training_set = dataset_train.iloc[:, 1:2].values <br><br>

# Feature Scaling <br>
from sklearn.preprocessing import MinMaxScaler<br>
sc = MinMaxScaler(feature_range = (0, 1))<br>
training_set_scaled = sc.fit_transform(training_set)<br><br>

# Creating a data structure with 60 timesteps and 1 output <br> #You can adjust timesteps
X_train = [] <br>
y_train = [] <br>
for i in range(60, 1258): <br>
    X_train.append(training_set_scaled[i-60:i, 0]) <br>
    y_train.append(training_set_scaled[i, 0]) <br>
X_train, y_train = np.array(X_train), np.array(y_train) <br><br>

# Reshaping<br>
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))<br><br>

# Building the RNN by Importing the Keras libraries and packages <br>
from keras.models import Sequential<br>
from keras.layers import Dense <br>
from keras.layers import LSTM <br>
from keras.layers import Dropout <br><br>

# Initialising the RNN<br>
regressor = Sequential()<br><br>

# Adding the 1st LSTM layer and some Dropout regularisation<br>
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))<br>
regressor.add(Dropout(0.2))<br><br>

# Adding a 2nd LSTM layer and some Dropout regularisation<br>
regressor.add(LSTM(units = 50, return_sequences = True))<br>
regressor.add(Dropout(0.2))<br><br>

# Adding a 3rd LSTM layer and some Dropout regularisation<br>
regressor.add(LSTM(units = 50, return_sequences = True))<br>
regressor.add(Dropout(0.2))<br><br>

# Adding 4th LSTM layer and some Dropout regularisation<br>
regressor.add(LSTM(units = 50))<br>
regressor.add(Dropout(0.2))<br><br>

# Adding the output layer<br>
regressor.add(Dense(units = 1))<br><br>

# Compiling the RNN<br>
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')<br><br>

# Fitting the RNN to the Training set<br>
regressor.fit(X_train, y_train, epochs = 25, batch_size = 32)
<br>
![image](https://github.com/Kanangnut/Predicting-Stock-Market-Index-Using-LSTM-yFinance/assets/130201193/eb42d581-2fa4-4b0b-9311-601b42f075cd)
<br>
# Making the predictions and visualising the results By Getting the real and predicted stock price<br>
dataset_test = pd.read_csv('../Simulation-HFT-Sentiment-Analyze-From-yFinance-byCat/TSLA.csv')<br>
real_stock_price = dataset_test.iloc[:, 1:2].values<br>
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)<br>
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values<br>
inputs = inputs.reshape(-1,1)<br>
inputs = sc.transform(inputs)<br>
X_test = []<br>
for i in range(60, 3346): #we have 3286 in dataset so we will calculate the range from (60 + 3286 = 3346),<br>
    X_test.append(inputs[i-60:i, 0])<br>
X_test = np.array(X_test)<br>
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))<br>
predicted_stock_price = regressor.predict(X_test)<br>
predicted_stock_price = sc.inverse_transform(predicted_stock_price)<br>

![image](https://github.com/Kanangnut/Predicting-Stock-Market-Index-Using-LSTM-yFinance/assets/130201193/867b673e-88a5-43ac-a14d-785e2a85ae07)
<br>
# Visualising the results<br>
plt.figure(figsize=(12, 6)) <br>
plt.plot(real_stock_price, color='blue', linestyle='-', label='Real TSLA Stock Price') <br>
plt.plot(predicted_stock_price, color='red', linestyle='--', label='Predicted TSLA Stock Price') <br>
plt.title('TSLA Stock Price Prediction') <br>
plt.xlabel('Date')  <br>
plt.ylabel('TSLA Stock Price') <br>
plt.grid(True)  <br>
plt.legend(loc='upper left') <br>
plt.tight_layout() <br>
plt.show() <br>

![image](https://github.com/Kanangnut/Predicting-Stock-Market-Index-Using-LSTM-yFinance/assets/130201193/1dd05caa-a902-4a0c-9cfe-2e2adac9fcf8)


























