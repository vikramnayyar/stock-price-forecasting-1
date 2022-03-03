# Stock Market Forecasting

## Demo


## Introduction
An app is developed for forecasting stock prices. The app was trained using <b>LSTM</b> model and is developed in <b>Streamlit</b>. 


## Dataset
The dataset consists of actual stock prices, acquired using **Yahoo! finance** library. Historical stock prices of desired range can be acquired using this library. 

## Problem Statement
Numerous factors affect the rise of a property in a region. With the growth of real estate industry; many new regions have emerged. Therefore, there are several opportunities and manifold speculations and verdicts. Identifying a privileged property has become challenging.

Conventionally; customers tended to buy neaby properties. But, with globalisation and cultural convergence; customers are more open to newer locations. Presently, customers prefer premium property; that persistently develops. Property price reliably determines such development. This ensures a high standard of living for customers. 

As real-estate demands large investment, the project is very significant.


## Goal
This work was performed as a personal project and is based on the dataset available on Zillow. The motivation was to accurately forecast the American house prices. 

An app forecasting the house prices, provides an intuitive means for identifying suitable property. This provides a trust in customers decision and validates current house price. The app will be utilized by the customers seeeking real-estate purchase. 


## System Environment
![](https://forthebadge.com/images/badges/made-with-python.svg)



[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/e/ed/Pandas_logo.svg" width=200>](https://pandas.pydata.org/)     [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/512px-NumPy_logo_2020.svg.png" width=200>](https://numpy.org/)     



[<img target="_blank" src="https://www.metachris.com/images/posts/logzero/logo-text-wide-cropped.png" width=200>](https://pypi.org/project/logzero/)     [<img target="_blank" src="https://user-images.githubusercontent.com/965439/27257445-8791ea14-539c-11e7-8f5a-eec6cdfababa.png" width=200>](https://pypi.org/project/PyYAML/)     [<img target="_blank" src="https://phyblas.hinaboshi.com/rup/nayuki/2020/pythonsubprocess.png" width=200>](https://docs.python.org/3/library/subprocess.html)



[<img target="_blank" src="https://matplotlib.org/_static/logo2_compressed.svg" width=200>](https://matplotlib.org)     [<img target="_blank" src="https://www.askpython.com/wp-content/uploads/2020/11/ARIMA-model-with-python-1024x512.jpg.webp" width=200>](https://pypi.org/project/pmdarima/)     [<img target="_blank" src="https://www.statsmodels.org/stable/_images/statsmodels-logo-v2-horizontal.svg" width=200>](https://www.statsmodels.org/stable/index.html)      




[<img target="_blank" src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png" width=200>](https://streamlit.io/)     [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Heroku_logo.svg/2560px-Heroku_logo.svg.png" width=200>](https://www.heroku.com/)

## Technical Description
The main project scripts are in the **"src"** directory. Exceptionally, **"app.py"** is in app directory. The main constituting scripts are as follows

* **get_data.py:** The script downloads the dataset using google drive link. The file **data.csv** downloads in data directory. The dataset is read, analyzed, cleaned and saved as **"cleaned_data.csv"** in **data** directory. Futhermore, region-wise time-series data is extracted from the dataset and respectively saved in region subdirectory (withing data directory).  

* **data_analysis.py:** This script obtains various visualizations of the dataset. These visualizations are saved in the **"Visualization"** directory. 

* **prepare_data.py:** The script checks the stationarity, accordingly removes the trend and splits train and test data. Train data and test data are respectively saved as **"train_data.csv"** and **"test_data.csv"** (in the data directory).

* **model_data.py:** The script determines ARIMA parameters; and trains ARIMA; on train set. Using the model; forecast is obtained for the test set. After validating the forecast, trained ARIMA model is saved as **"model.pkl"**. 

* **app.py:** The script develops a Streamlit app; that accepts **40** American regions. Also, user selects the forecast duration (in years). The inputs are transformed and fed to **model.pkl**. Accordingly, the model's forecast is displayed in the application. 
 
* **run_project.py:** The script runs all the project scripts (discussed in this section) sequentially. Therefore, entire project is executed with this script.  

**get_data_util.py**, **data_analysis_util.py**, **prepare_data_util.py**, **model_data_util.py** and **utility.py** declare vital functions that are required by respective scripts. 

## Directory Structure

```bash
├── app                              # Application files
|  ├── app.py                        # Application script
├── config                           # Configuration files
|  ├── config.yaml                   # Configuration file  
├── data                             # Data files 
|  ├── data.csv                      # Original dataset, that downloads from google drive (Not present in repository)
|  ├── clean_data.csv                # Cleaned dataset 
|  ├── prepared_data.csv             # Prepared dataset 
|  ├── train_data.csv                # Train data
|  ├── test_data.csv                 # Test data
|  |  ├── region                     # Subdirectory that contains region-wise time-series data
├── log                              # Log files
|  ├── get_data.log                  # "get_data.py" script logs
|  ├── data_analysis.log             # "data_analysis.py" script logs
|  ├── prepare_data.log              # "prepare_data.py" script logs 
|  ├── model_data.log                # "model_data.py" script logs 
├── model                            # Model Files
|  ├── model.pkl                     # Saved model
├── src                              # Main project scripts 
|  ├── get_data.py                   # Dataset acquistion and cleaning script
|  ├── get_data_util.py              # script declaring utility functions for get_data.py 
|  ├── data_analysis.py              # Dataset analysis and visualization script
|  ├── data_analysis_util.py         # script declaring utility functions for data_analysis.py
|  ├── prepare_data.py               # Dataset preperation script
|  ├── prepare_data_util.py          # script declaring utility functions for prepare_data.py
|  ├── model_data.py                 # Dataset modelling script
|  ├── model_data_util.py            # script declaring utility functions for model_data.py
|  ├── utility.py                    # script declaring general utility functions
├── visualizations                   # Dataset visualizations
|  ├── arima_diagnosis.png           # Age vs deposit figure
|  ├── decompose_series.png          # Balance vs deposit figure
|  ├── house_prices.png              # Education vs deposit figure
|  ├── moving_average.png            # Job vs deposit figure 
|  ├── price_density.png        # Marital vs deposit figure
|  ├── price_forecast.png           # Dataset balance figure
|  ├── stationarity_test.png       # Correalation heatmap of features
|  ├── train-test-series.png        # Feature importance of best model
├── LICENSE                          # License
├── Procfile                         # Required for Heroku deployment 
├── README.md                        # Repository description
├── requirements.txt                 # Required libraries
├── setup.sh                         # Required for Heroku deployment

```

## Installing Dependencies
Foremost running the project, installing the dependencies is essential. 
* Ensure Python 3.8.8 or later is installed in the system. 
* All required libraries are listed in "requirements.txt". These are easily installed; by running the following command in project directory
```bash
pip install -r requirements.txt
```

## Run Project
As discussed in **Technical Aspect** section, "src" and “app” directory respectively possess the main scripts application scripts. 

Running the following command in the "src" directory executes the entire project  
```bash
python3 run_project.py
```
Alternatively, a project script can be individually executed using the general script 
```bash
python3 script.py
```
Here “script.py” represents any project script. 

Exceptionally, application file "app.py" runs using command 
```bash
streamlit run app/app.py
```
**Note:** To run any project script, directory location must be correct.
