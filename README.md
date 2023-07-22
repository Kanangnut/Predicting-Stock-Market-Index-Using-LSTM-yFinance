<b>Stock Price Prediction of Tesla Inc Stock by Long Short-Term Memory(LSTM), Recurrent Neural Network(RNN)</b><br>
This project using TSLA stock from 2010 to 2023<br>

Dataset:<br>
The dataset is taken from yahoo website in CSV format.The dataset consists of Open, High, Low, Close, Adj Close Prices and Valume of Tesla Inc. <br>
stocks from 29th Jun 2010 to 19th Junly 2023, all 3287 rows.<br>

Indicator:
Stock traders mainly use Open price for predicttion.

Processing and Model:
Long short-term memory(LSTM) network is a recurrent neural network(RNN), aimed to deal with the vanishing gradient problem present in traditional RNNs.
RNN is used in this code to predict stock prices. The program begins by importing the training set from a CSV file, specifically the 'TSLA.CSV'. Then subjected to feature scaling using the MinMaxScaler from sklearn preprocessing to normalize the data and improve performance of the network. Then creating sequences with 60 timesteps, it represents past stock prices and their corresponding output, which is the next stock price.Then Keras is used to build the RNN model with four LSTM layers stacked to runing the price stock data.

Version:
Python 3.11.1 in Visual Studio Code

Visualising the results:





















first step pls install program:
1. pip install numpy
2. pip install matplotlib
3. pip install yfinance
4. pip install pandas

![image](https://github.com/Kanangnut/Predicting-Stock-Market-Index-Using-LSTM-yFinance/assets/130201193/1dd05caa-a902-4a0c-9cfe-2e2adac9fcf8)


























