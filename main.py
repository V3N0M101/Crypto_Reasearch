import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from statsmodels.tsa.statespace.sarimax import SARIMAX

def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=False, date_format='%m/%d/%Y')
    data.sort_values('Date', inplace=True)
    data['Price'] = data['Price'].astype(str).str.replace(',', '').astype(float)
    data['Change %'] = data['Change %'].str.replace('%', '').astype(float)
    return data

def plot_trend(data, x, y, xlabel, ylabel, title, label, color='blue'):
    plt.plot(data[x], data[y], label=label, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)


def create_and_show_plots(data, label, prediction_period=365):
    data['Date'] = pd.to_datetime(data['Date'])
    data_filtered = data[(data['Date'] >= '2017-01-01') & (data['Date'] <= '2024-07-15')]

    plot_data = {
        'Price Trend': {'y': 'Price', 'color': 'blue'},
        'Daily Price Changes': {'y': 'Change %', 'color': 'green'},
        'Price Trend with Moving Averages': {'y': 'Price', 'ma': True},
        'Price Changes in 2024': {'y': 'Price', 'start_date': '2024-01-01'}
    }

    for title, params in plot_data.items():
        fig = plt.figure(figsize=(10, 6))
        fig.canvas.manager.set_window_title(f'{label} {title}')
        if 'ma' in params:
            data['7-day MA'] = data['Price'].rolling(window=7).mean()
            data['30-day MA'] = data['Price'].rolling(window=30).mean()
            plt.plot(data['Date'], data['Price'], label='Price', color='blue')
            plt.plot(data['Date'], data['7-day MA'], label='7-day MA', color='orange')
            plt.plot(data['Date'], data['30-day MA'], label='30-day MA', color='red')
        else:
            plot_data = data_filtered if 'start_date' not in params else data[
                (data['Date'] >= params['start_date']) & (data['Date'] <= '2024-07-15')]
            plot_trend(plot_data, 'Date', params['y'], 'Date', f'{params["y"]} (USD)', f'{label} {title}', params['y'],
                       color=params.get('color', 'blue'))

    # Prophet model
    prophet_data = data_filtered[['Date', 'Price']].copy()  # Copy the filtered data
    prophet_data.rename(columns={'Date': 'ds', 'Price': 'y'},
                        inplace=True)  # Prophet requires 'ds' for Date and 'y' for the value

    # Cap and floor values
    prophet_data['cap'] = prophet_data['y'].max() * 1.1  # Cap at 110% of max
    prophet_data['floor'] = prophet_data['y'].min() * 0.9  # Floor at 90% of min

    # Create and configure the Prophet model
    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,  # Daily seasonality off unless needed
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05,
        interval_width=0.95
    )

    # Add custom monthly seasonality if needed
    prophet_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    # Fit the model
    prophet_model.fit(prophet_data)

    # Make future predictions
    future = prophet_model.make_future_dataframe(periods=prediction_period)
    future['cap'] = prophet_data['cap'].max()  # Apply the cap for future data
    future['floor'] = prophet_data['floor'].min()  # Apply the floor for future data

    # Predict using Prophet
    forecast = prophet_model.predict(future)

    # Plot the forecast
    fig5 = plt.figure(figsize=(10, 6))
    fig5.canvas.manager.set_window_title(f'{label} Price Forecast with Prophet')

    # Plot historical prices
    plt.plot(prophet_data['ds'], prophet_data['y'], label='Historical Price', color='blue')

    # Plot predicted prices
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Price (Prophet)', color='green')

    # Fill between the confidence intervals
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.3,
                     label='Confidence Interval')

    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)

    fig6 = plt.figure(figsize=(10, 6))
    fig6.canvas.manager.set_window_title(f'{label} Price Prediction with Prophet')

    # Plot the historical prices
    plt.plot(prophet_data['ds'], prophet_data['y'], label='Historical Price', color='blue')

    # Plot the predicted prices from Prophet
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Price (Prophet)', color='green')

    # Fill between the confidence intervals
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.3,
                     label='Confidence Interval')

    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)

    # ARIMA model
    arima_data = data_filtered.set_index('Date')['Price']
    arima_model = ARIMA(arima_data, order=(5, 1, 1))
    arima_fit = arima_model.fit()
    arima_forecast = arima_fit.get_forecast(steps=prediction_period)
    forecast_index = pd.date_range(start='2024-07-15', periods=prediction_period)
    arima_series = arima_forecast.predicted_mean
    arima_ci = arima_forecast.conf_int()

    fig7 = plt.figure(figsize=(10, 6))
    fig7.canvas.manager.set_window_title(f'{label} Future Data (ARIMA)')
    plt.plot(arima_data.index, arima_data, label='Historical Price', color='blue')
    plt.plot(forecast_index, arima_series, label='Forecasted Price', color='red')
    plt.fill_between(forecast_index, arima_ci.iloc[:, 0], arima_ci.iloc[:, 1], color='pink', alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)

    # LSTM model
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_filtered['Price'].values.reshape(-1, 1))

    def create_lstm_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 60
    X, y = create_lstm_dataset(data_scaled, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    train_size = int(len(X) * 0.8)
    trainX, testX = X[:train_size], X[train_size:]
    trainY, testY = y[:train_size], y[train_size:]

    # Optimized LSTM Model
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=100, return_sequences=True, input_shape=(time_step, 1)))
    lstm_model.add(Dropout(0.2))  # Dropout rate of 20%
    lstm_model.add(LSTM(units=100))
    lstm_model.add(Dropout(0.2))  # Dropout rate of 20%
    lstm_model.add(Dense(1))

    # Compile model with Adam optimizer and learning rate 0.001
    adam = Adam(learning_rate=0.001)
    lstm_model.compile(optimizer=adam, loss='mean_squared_error')

    # Train the model
    lstm_model.fit(trainX, trainY, epochs=50, batch_size=32, validation_data=(testX, testY))

    # Predict on train and test sets
    trainPredict = scaler.inverse_transform(lstm_model.predict(trainX))
    testPredict = scaler.inverse_transform(lstm_model.predict(testX))
    train_dates = data_filtered['Date'].iloc[time_step:time_step + len(trainPredict)]
    test_dates = data_filtered['Date'].iloc[time_step + len(trainPredict):time_step + len(trainPredict) + len(testPredict)]

    fig8 = plt.figure(figsize=(10, 6))
    fig8.canvas.manager.set_window_title(f'{label} Price Prediction with LSTM')
    plt.plot(data_filtered['Date'], data_filtered['Price'], label='Historical Price', color='blue')
    plt.plot(train_dates, trainPredict.flatten(), label='Train Predicted Price', color='green')
    plt.plot(test_dates, testPredict.flatten(), label='Test Predicted Price', color='purple')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)

    # Predict future prices
    n_future = prediction_period  # Number of days to predict
    last_sequence = data_scaled[-time_step:]  # Last available sequence

    future_predictions = []
    current_sequence = last_sequence.reshape((1, time_step, 1))  # Ensure current_sequence is 3D

    for _ in range(n_future):
        next_price = lstm_model.predict(current_sequence)[0, 0]
        future_predictions.append(next_price)
        # Append the new prediction and remove the oldest entry
        current_sequence = np.append(current_sequence[:, 1:, :], [[[next_price]]], axis=1)

    # Inverse transform predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Create date range for future predictions
    last_date = data_filtered['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=n_future + 1)[1:]  # Exclude the start date

    fig9 = plt.figure(figsize=(10, 6))
    fig9.canvas.manager.set_window_title(f'{label} Future Price Prediction with LSTM')
    plt.plot(data_filtered['Date'], data_filtered['Price'], label='Historical Price', color='blue')
    plt.plot(future_dates, future_predictions, label='Future Predicted Price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)

    # SARIMA model
    sarima_data = data_filtered.set_index('Date')['Price']
    sarima_model = SARIMAX(sarima_data, order=(5, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_fit = sarima_model.fit()
    sarima_forecast = sarima_fit.get_forecast(steps=prediction_period)
    sarima_series = sarima_forecast.predicted_mean
    sarima_ci = sarima_forecast.conf_int()

    fig10 = plt.figure(figsize=(10, 6))
    fig10.canvas.manager.set_window_title(f'{label} Future Data (SARIMA)')
    plt.plot(sarima_data.index, sarima_data, label='Historical Price', color='blue')
    plt.plot(forecast_index, sarima_series, label='Forecasted Price', color='red')
    plt.fill_between(forecast_index, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1], color='pink', alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)

    plt.show()

def BTC(prediction_period):
    return create_and_show_plots(btc_data, 'Bitcoin', prediction_period=prediction_period)

def ETH(prediction_period):
    return create_and_show_plots(eth_data, 'Ethereum', prediction_period=prediction_period)

def LTC(prediction_period):
    return create_and_show_plots(ltc_data, 'Litecoin', prediction_period=prediction_period)

def main():
    print("BTC | ETH | LTC")
    crypto = input("Which one: ").upper()
    prediction_period = int(input("Enter the number of days to predict: "))
    if crypto in ['ETH', 'BTC', 'LTC']:
        if crypto == 'ETH':
            ETH(prediction_period)
        elif crypto == 'LTC':
            LTC(prediction_period)
        elif crypto == 'BTC':
            BTC(prediction_period)
    else:
        raise ValueError("Please enter a valid cryptocurrency")

if __name__ == '__main__':
    btc_data = load_data('Bitcoin Historical Data.csv')
    eth_data = load_data('Ethereum Historical Data.csv')
    ltc_data = load_data('Litecoin Historical Data.csv')
    main()

