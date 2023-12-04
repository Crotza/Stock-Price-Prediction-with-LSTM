# Stock Price Prediction with LSTM

This project utilizes Long Short-Term Memory (LSTM) neural networks to predict stock prices for various companies. It uses historical stock price data to train an LSTM model and then makes predictions for the future stock prices.

## Dependencies
To run this project, you'll need to install the following dependencies:

- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [yfinance](https://pypi.org/project/yfinance/)
- [TensorFlow](https://www.tensorflow.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)

You can install them using `pip` or any other package manager.

```bash
pip install numpy pandas yfinance tensorflow scikit-learn matplotlib

**Global Definitions**

You can customize the behavior of the project by changing the global definitions at the beginning of the script:

- `START_DATE`: Start date for historical data.
- `END_DATE`: End date for historical data (set to yesterday by default).
- `EPOCHS`: Number of training epochs for the LSTM model.
- `BATCH_SIZE`: Batch size for training.
- `SEQUENCE_LENGTH`: Number of days used for prediction.
- `SPLIT_RATIO`: Ratio for splitting the data into training and testing sets.

**How to Run**

To run the project, execute the Python script in your terminal or preferred Python environment. It will provide you with options to predict stock prices for predefined companies and allow you to enter a custom company symbol for prediction.

```bash
python stock_price_prediction.py

**Predefined Companies**

The script comes with predefined companies for which you can predict stock prices. These companies include:

- Petrobras (PBR)
- Vale (VALE)
- Itaú Unibanco (ITUB)
- Banco Bradesco (BBD)
- Ambev (ABEV)

The script will automatically generate predictions for these companies.

**User Input**

After predicting prices for predefined companies, the script will ask if you would like to predict prices for an additional company. You can enter the company name and stock symbol, and the script will provide predictions for that company as well.

**Results**

The script will display the last week's stock prices for the selected company, make predictions for the next 7 days, calculate the Root Mean Squared Error (RMSE) for the predictions, and visualize the predicted vs. actual stock prices in a graph.

Feel free to explore and use this project to predict stock prices for different companies of interest.
