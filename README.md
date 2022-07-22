# LSTM Stock Predictor

![deep-learning.jpg](Images/deep-learning.jpg)

Due to the volatility of cryptocurrency speculation, investors will often try to incorporate sentiment from social media and news articles to help guide their trading strategies. One such indicator is the [Crypto Fear and Greed Index (FNG)](https://alternative.me/crypto/fear-and-greed-index/) which attempts to use a variety of data sources to produce a daily FNG value for cryptocurrency. You have been asked to help build and evaluate deep learning models using both the FNG values and simple closing prices to determine if the FNG indicator provides a better signal for cryptocurrencies than the normal closing price data.

In this assignment, you will use deep learning recurrent neural networks to model bitcoin closing prices. One model will use the FNG indicators to predict the closing price while the second model will use a window of closing prices to predict the nth closing price.

You will need to:

1. [Prepare the data for training and testing](#prepare-the-data-for-training-and-testing)
2. [Build and train custom LSTM RNNs](#build-and-train-custom-lstm-rnns)
3. [Evaluate the performance of each model](#evaluate-the-performance-of-each-model)

- - -

## Files

[Closing Prices Starter Notebook](Starter_Code/lstm_stock_predictor_closing.ipynb)

[FNG Starter Notebook](Starter_Code/lstm_stock_predictor_fng.ipynb)

- - -

## Instructions

### Prepare the data for training and testing

### Use the starter code as a guide to create a Jupyter Notebook for each RNN. The starter code contains a function to create the window of time for the data in each dataset

* > [Closing Prices Starter Notebook](Starter_Code/lstm_stock_predictor_closing.ipynb)

### For the Fear and Greed model, you will use the FNG values to try and predict the closing price. A function is provided in the notebook to help with this

* > [FNG Starter Notebook](Starter_Code/lstm_stock_predictor_fng.ipynb)

### For the closing price model, you will use previous closing prices to try and predict the next closing price. A function is provided in the notebook to help with this

* > I created a DataFrame of Real and Predicted values:

### Code 

* > stocks = pd.DataFrame({
    "Real": real_prices.ravel(),
    "Predicted": predicted_prices.ravel()
}, index = df.index[-len(real_prices): ])
stocks.head()

### Each model will need to use 70% of the data for training and 30% of the data for testing

* > For each model, 70% of the data was used for training and the rest for testing.

### Code 

split = int(0.7 * len(X))
X_train = X[: split]
X_test = X[split:]
y_train = y[: split]
y_test = y[split:]

Apply a MinMaxScaler to the X and y values to scale the data for the model.

* > I imported sklearn.preprocessing and MinMaxScaler and scaled the data between 0 and 1 Use the MinMaxScaler to scale data between 0 and 1

### Code

* > scaler = MinMaxScaler()

Finally, reshape the X_train and X_test values to fit the model's requirement of samples, time steps, and features. (*example:* `X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))`)

* > I reshaped the features for the model

### Code  

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
print (f"X_train sample values:\n{X_train[:5]} \n")
print (f"X_test sample values:\n{X_test[:5]}")

### Build and train custom LSTM RNNs

In each Jupyter Notebook, create the same custom LSTM RNN architecture. In one notebook, you will fit the data using the FNG values. In the second notebook, you will fit the data using only closing prices.

Use the same parameters and training steps for each model. This is necessary to compare each model accurately.

### Evaluate the performance of each model

Finally, use the testing data to evaluate each model and compare the performance.

Use the above to answer the following:

## Which model has a lower loss?

* > Based on the verbose output of the training data which includes the loss and accuracy of the model, FGN had a higher accuracy rate of %25 (closing prices %15) when the window size of both models is 5.

### Output

* > Performance Model: Window of closing prices

5/5 [==============================] - 1s 5ms/step - loss: 0.1585
0.15848222374916077

## Output 

* > Performance Model: FGN Indicators

6/6 [==============================] - 1s 3ms/step - loss: 0.2593
0.25925248861312866

## Which model tracks the actual values better over time?

* > As you can see from the charts below, neither model has been tracked relatively efficiently. Closing prices followed more closely, whatever the size of the window.

![Window of closing prices](https://github.com/JamelBoyer/14-Homework/blob/main/14-Homework/Deep%20Learning/Images/Window.jpg?raw=true)

![FGN Indicators](https://github.com/JamelBoyer/14-Homework/blob/main/14-Homework/Deep%20Learning/Images/FGN.jpg?raw=true)

## Which window size works best for the model?

* > With the closing prices model lower window sizes tracked better overtime. Below is an example of wwindow size 1(see below).

![Window of closing prices!](https://github.com/JamelBoyer/14-Homework/blob/main/Images/ClosingPrices.jpg)

* > Same pheonomen with the FNG Model. Below is a plot of window size 2.

![FNG Model](https://github.com/JamelBoyer/14-Homework/blob/main/14-Homework/Deep%20Learning/Images/Window_size_2.jpg?raw=true)

### Resources

[Keras Sequential Model Guide](https://keras.io/getting-started/sequential-model-guide/)

[Illustrated Guide to LSTMs](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)

[Stanford's RNN Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)
