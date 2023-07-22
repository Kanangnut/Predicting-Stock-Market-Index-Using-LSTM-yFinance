<b>Stock Price Prediction of Tesla Inc Stock by Long Short-Term Memory (LSTM), Recurrent Neural Network (RNN)</b><br>
This project using TSLA stock from 2010 to 2023<br>

<b>Dataset:</b><br>
The dataset is taken from yahoo website in CSV format.The dataset consists of Open, High, Low, Close, Adj Close Prices and Valume of Tesla Inc. <br>
stocks from 29th Jun 2010 to 19th Junly 2023, all 3286 rows.<br>

<b>Indicator:</b><br>
Stock traders mainly use Open price for prediction.

<b>Processing and Model:</b><br>
Long short-term memory(LSTM) network is a recurrent neural network(RNN), aimed to deal with the vanishing gradient problem present in traditional RNNs.
RNN is used in this code to predict stock prices. The program begins by importing the training set from a CSV file, specifically the 'TSLA.CSV'. Then subjected to feature scaling using the MinMaxScaler from sklearn preprocessing to normalize the data and improve performance of the network. Then creating sequences with 60 timesteps, it represents past stock prices and their corresponding output, which is the next stock price.Then Keras is used to build the RNN model with four LSTM layers stacked to runing the price stock data.

<b>Version:</b><br>
Python 3.11.2 in Visual Studio Code

<b>Visualising the results:</b><br>
The model can predict the trend of the actual stock prices quite closely. 

<b>Observation and Conclusion:</b><br>
From comparison of training result the accuracy of the model can be enhanced by training with more data and increasing the LSTM layers.<br><br>
Result of 1006 dataset<br> 
![image](https://github.com/Kanangnut/Predicting-Stock-Using-LSTM-Neural-networks-yFinance/assets/130201193/d46328ff-ae93-4b8f-9aaf-f49abfb602af)

Result of 3286 dataset<br> 
![image](https://github.com/Kanangnut/Predicting-Stock-Using-LSTM-Neural-networks-yFinance/assets/130201193/ec3628f8-4e5e-439d-a73a-2436c8d26b68)


























