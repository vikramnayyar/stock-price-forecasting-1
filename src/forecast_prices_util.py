"""

The script declare the functions used in get_data.py 

"""

import pathlib
from logzero import logger, logfile
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


def create_log(log_name):
    tgt_path = pathlib.Path.cwd().parent.joinpath('log')
    logfile(tgt_path/log_name)     # Creating logfile


def analyze_data(df): 
    logger.info('\n * Size of dataframe: {}\n'.format(df.shape))
    logger.info('* Column-wise NaNs can be identified as: ')
    logger.info('{}\n'.format(df.isnull().sum()))
    logger.info('Total NaNs:{}'.format(df.isnull().sum().sum()))



def plot_prices(df):
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Date')
    plt.ylabel('Close Prices')
    plt.plot(df['Close'])
    plt.title('Historical Price of Cadila Healthcare')
    plt.savefig("../visualizations/historical_price.png")
    plt.close()


def create_dataset(dataset, time_step):      # time_step = 1 changed
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):    # To allow validation, i cannot not exceed len(df) - time_step
        a = dataset[i: (i+time_step), 0]         # Dataset is created, from i to time_step. This is input for prediction   

        dataX.append(a)                          # append input to prediction to dataX
        dataY.append(dataset[i + time_step, 0])  # append the value (single) after time step to dataY. This is output 

    return np.array(dataX), np.array(dataY)  


def create_model():
    model = Sequential()      # Declaring sequential layer
    model.add(LSTM(50, return_sequences = True, input_shape = (100, 1) ))  # First LSTM layer--- Input must be equal to input step
    model.add(LSTM(50, return_sequences = True))   # Second LSTM layer
    model.add(LSTM(50))     # 
    
    model.add(Dense(1))    # Final layer to obtain output 
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')   # Assignign loss metric and optimizer to the model
    
    logger.info("The model of following config was created: \n {}" .format(model.summary()))

    return model