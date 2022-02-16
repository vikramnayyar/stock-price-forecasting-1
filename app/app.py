"""
The script develops an application interface. Using app the user selects a stock.
The selected stock's historical stock prices are scraped. The trained model is imported 
and future stock prices are forecasted.
"""

import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date


#------------------------ Streamlit GUI
st.title('Nifty 50 Stocks Forecasting')

st.write("For coming 30 days, identify the stocks; that will return largest profits! ")
 
# st.write("Assure good selection; by testing the stocks recommended by experts in moneycontrol or economictimes")


st.write("HAPPY INVESTING !!")

#------------------------- Selectbox for selecting scrip
scrip_id = 'TVSMOTOR.NS'

#stock_list = ['ULTRACEMCO.NS', 'TCS.NS', 'KOTAKBANK.NS', 'ONGC.NS', "LT.NS", 
#              "JSWSTEEL.NS", "HINDUNILVR.NS", "TATAMOTORS.NS", "TITAN.NS", "ASIANPAINTS.NS", "UPL.NS", "SUNPHARMA.NS"]   # sector-wise stocks

stock_list = ["ADANIPORTS.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS",
              "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS",
              "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS",
              "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "HDFC.NS", "ICICIBANK.NS",
              "ITC.NS", "IOC.NS", "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
              "M&M.NS", "MARUTI.NS", "NTPC.NS", "NESTLEIND.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS",
              "SBILIFE.NS", "SHREECEM.NS", "SBIN.NS", "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", 
              "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "UPL.NS", "ULTRACEMCO.NS", 
              "WIPRO.NS"]


# stock_list = ['BALRAMCHIN.NS', 'CADILAHC.NS', 'RAMCOCEM.NS', 'TVSMOTOR.NS']
scrip_id = st.selectbox('Select the stock', stock_list)
st.write('You selected:', scrip_id)


#------------------------- Scraping stock prices
today = date.today()

df = yf.download(scrip_id,
                 start = '2000-09-1',   # 2000-09-19
                 end = today)   # end = '2021-06-30'

df.to_csv('data/stock_data_streamlit.csv')



#------------------- Reading Model
from keras.models import model_from_json

# load json and create model
json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("model/model.h5")
print("Loaded model from disk")



# -------------------- Reading Data
df = pd.read_csv('data/stock_data_streamlit.csv')    
df_close =  df.Close

# ------------------- Preprocessing (Standardizing) stock prices
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
df_close = scaler.fit_transform(np.array(df_close).reshape(-1,1))


#--------------------- Transforming Dataset According to Time Step
import numpy
def create_dataset(dataset, time_step):      # time_step = 1 changed
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):    # To allow validation, i cannot not exceed len(df) - time_step
        a = dataset[i: (i+time_step), 0]         # Dataset is created, from i to time_step. This is input for prediction   

        dataX.append(a)                          # append input to prediction to dataX
        dataY.append(dataset[i + time_step, 0])  # append the value (single) after time step to dataY. This is output 

        
    return numpy.array(dataX), numpy.array(dataY)  


time_step = 100  # Hyperparameter 1
    
train_set, train_label = create_dataset(df_close, time_step)


# ----------------------Predicting Next 30 Days


x_diff = len(df_close) - 100   

x_input = df_close[x_diff:].reshape(1,-1)
x_input.shape


temp_input = list(x_input)
temp_input = temp_input[0].tolist()



# demonstrate prediction for next 10 days

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



#-------------- Plotting Forecast

# Previously predicted days (100 previous days)
day_new = np.arange(1, 101)

# Future days to predict
day_pred = np.arange(101, 131)

adj_len = df_close.shape[0] - 100


import matplotlib.pyplot as plt

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.plot(day_new, scaler.inverse_transform(df_close[adj_len:]), label = "Price History")

ax.plot(day_pred, scaler.inverse_transform(lst_output), label = "Price Forecast")
ax.legend()
ax.set_title(scrip_id)
ax.set_xlabel("Days")
ax.set_ylabel("Price")

st.write(fig)
