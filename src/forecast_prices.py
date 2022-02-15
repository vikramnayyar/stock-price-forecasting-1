"""

The script reads and analyzes the dataset. After reading, it is oberved that
dataset cleaning is not required.

Also, the **region-wise** datasets are extecated and saved. This is 
required for the application to accept user inputs. 

"""
#%% importing modules
import pandas as pd
from logzero import logger

import numpy as np
import math

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from forecast_prices_util import analyze_data, plot_prices, create_dataset, create_model, create_log

create_log("forecast_prices.log")  # Creating log file

#%%
##################################################
#-------------------Reading Data------------------
##################################################

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
df = pd.read_csv('../data/cadilahc.csv', index_col='Date', parse_dates=['Date'], date_parser=dateparse)    

df = df.drop(["Adj Close"], axis = 1)


#%%
##################################################
#------------------Analyzing Data-----------------
##################################################

analyze_data(df)

plot_prices(df)


#%%
##################################################
#---------------Standardizing Data----------------
##################################################

scaler = MinMaxScaler(feature_range = (0,1))
df_close =  df["Close"]

scaler = MinMaxScaler(feature_range = (0,1))
df_close = scaler.fit_transform(np.array(df_close).reshape(-1,1))


#%% Splitting data

# Splitting Train and Test data
train_size = int((len(df_close)*0.65))
test_size = len(df_close) - train_size
train_data, test_data = df_close[0: train_size, :], df_close[train_size: len(df_close), :1]


# Creating Training and Test Datasets (Based on Time Step) 

time_step = 100  # Hyperparameter 1
    
train_set, train_label = create_dataset(train_data, time_step)
test_set, test_label = create_dataset(test_data, time_step)

logger.info("Size of train set: {}".format(train_set.shape))
logger.info("Size of train label: {}".format(train_label.shape))
logger.info("Size of test set: {}".format(test_set.shape))
logger.info("Size of test label: {}".format(test_label.shape))

#%% Creating Stacked LSTM

# reshaping train and test sets according to LSTM
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1], 1)
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1], 1)

model = create_model()


#%% training model
model.fit(train_set, train_label, validation_data = (test_set, test_label), epochs = 10, batch_size = 64, verbose = 1)  # verbose displays the epochs


#%% Forecasting train_set and test_set

train_predict = model.predict(train_set)
test_predict = model.predict(test_set)


#%% Transform to original form

train_predict= scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

#%% Calculate RMSE Performance

# Train data RMSE
logger.info("Root Mean squared error of train set: {}".format(math.sqrt(mean_squared_error(train_label, train_predict))))

# Test data RMSE
logger.info("Root Mean squared error of test set: {}".format(math.sqrt(mean_squared_error(test_label, test_predict))))


#%% Plotting and comparing forecast
# shift train predictions for plotting

def plot_forecast(train_predict, test_predict, df_close):
    
    import matplotlib.pyplot as plt
    
    look_back = 100    # 100 is time_step size
    trainPredictPlot = np.empty_like(df_close)  # empty_like creates a df of same data type and dimension as df_close (Values are not same) 
    trainPredictPlot[:, :] = np.nan         # replacing df values with NaNs  
    
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict  # converting train_predict to plottable format 
    
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df_close)
    testPredictPlot[:, :] = np.nan
    
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df_close)-1, :] = test_predict
    
    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)
    
    ax.plot(scaler.inverse_transform(df_close), label = "Stock Price")   # Plotting stock prices 
    
    ax.plot(trainPredictPlot, label = "Train Set Forecast")   # Plotting train set prediction
    ax.plot(testPredictPlot, label = "Test Set Forecast")    # Plotting test set prediction
    
    ax.legend()
    ax.set_title("Cadilla Healthcare")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    
    plt.savefig("../visualizations/forecast_comparison.png")
    plt.close()

plot_forecast(train_predict, test_predict, df_close)

#%% Forecating next 30 days

x_diff = len(test_data) - 100    
x_input = test_data[x_diff:].reshape(1,-1)

temp_input = list(x_input)
temp_input = temp_input[0].tolist()

lst_output=[]
n_steps=100
i=0

while(i<30):       # i< 30 is used to predict next 30 days
    
    if(len(temp_input)>100):
        #print(temp_input)

        # shifting input by '1'
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))

        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]

        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    
#     First else condition executes
    else:
#     Reshaping input for LSTM
        x_input = x_input.reshape((1, n_steps,1))

#     Predicting future
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])

#     Adding prediction to input
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))

#     Adding prediction to the output
        lst_output.extend(yhat.tolist())

        i=i+1
#    As the temp_input is greater than 100 now, if statement will execute onwards

print(lst_output)

#%% plotting forecast

# Previously predicted days (100 previous days)
day_new = np.arange(1, 101)

# Future days to predict
day_pred = np.arange(101, 131)
adj_len = df_close.shape[0] - 100


fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(day_new, scaler.inverse_transform(df_close[adj_len:]), label = "Price History")
ax.plot(day_pred, scaler.inverse_transform(lst_output), label = "Price Forecast")

ax.legend()
ax.set_title("Cadilla Healthcare")
ax.set_xlabel("Days")
ax.set_ylabel("Price")

plt.savefig("../visualizations/30_day_forecast.png")

#%% Saving Model

# Serializing model to json
model_json = model.to_json()
with open("../model/model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("../model/model.h5")
print("Saved model to disk")

