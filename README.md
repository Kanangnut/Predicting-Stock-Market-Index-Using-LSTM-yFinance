<b>Stock Price Prediction of Tesla Inc Stock by Long Short-Term Memory (LSTM), Recurrent Neural Network (RNN)</b><br>
This project using TSLA stock from 2010 to 2023.<br>

<b>Dataset:</b><br>
The dataset is taken from yahoo website in CSV format.The dataset consists of Open, High, Low, Close, Adj Close Prices and Valume of Tesla Inc.
stocks from 29th Jun 2010 to 19th Junly 2023, all 3286 rows.<br>

<b>Indicator:</b><br>
Stock traders mainly use Open price for prediction.

<b>Processing and Model:</b><br>
Long short-term memory(LSTM) network is a recurrent neural network(RNN), aimed to deal with the vanishing gradient problem present in traditional RNNs.
RNN is used in this code to predict stock prices. The program begins by importing the training set from a CSV file, specifically the 'TSLA.CSV'. Then subjected to feature scaling using the MinMaxScaler from sklearn preprocessing to normalize the data and improve performance of the network. Then creating 4th sequences with 60 timesteps, it represents past stock prices and their corresponding output, which is the next stock price.Then Keras is used to build the RNN model with four LSTM layers stacked to runing the price stock data. At first simulation run the model with 2nd sequences but the accurancy of result decrease so decided to use the highest accuracy for this project.

<b>Version:</b><br>
Python 3.11.2 in Visual Studio Code.

<b>Visualising the results:</b><br>
The model can predict the trend of the actual stock prices quite closely. 

<b>Observation and Conclusion:</b><br>
From comparison of training result the accuracy of the model can be enhanced by training with more data and increasing the LSTM layers at 60 timesteps.<br><br>
Result of 1006 dataset.<br> 


Result of 3286 dataset.<br> 
![image](https://github.com/Kanangnut/Predicting-Stock-Using-LSTM-RNN-yFinance/assets/130201193/017116a9-1a8f-45f9-90d0-2e5ece8e0083)



























