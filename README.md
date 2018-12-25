# Deep-Learning-Applications-in-Finance

This readme file is included to cover the scope of all my work done so far as a part of Machine Learning with Networks Course Project and to summarize all the results obtained with it.

### DataSet Preparation
The data consists of two datasets; The Stock Price data and the fundamentals data. The Stock Price dataset was obtained from Yahoo Finance while the Fundamentals dataset was obtained from an online source (stockpups.com). The Stock Price Dataset consists of daily Stock Price data. The four relevant price measurements used in the paper were 'Open', 'High', 'Low' and 'Close'. 'High' is the highest trading price of the security on the given day. 'Open' is price at which the security trades when the market opens. 'Low' is the lowest trading price of a security on the given day. 'Close' is the closing price of the security on the given day. This paper focuses on prediction of securities in the Oil & Gas sector.  
The Fundamentals dataset consists of quarterly data from Balance Sheet, Cash Flow, and Income statements of Companies. The dataset compiled on stockpups.com, was obtained from XBRL filings and filings with the US securities and exchange commission.  Some of the important fundamental factors used in the model are given below:

- P/E ratio - The ratio of Price to EPS diluted TTM as of the previous quarter.
- P/B ratio - The ratio of Price to Book value of equity per share as of the previous quarter.
- Book Value of Equity per Share (BVPS) - Common stockholders' equity per share.
- Dividend Payout Ratio - The ratio of Dividends TTM to Earnings (available to common stockholders) TTM.
- Long-term debt to equity ratio - The ratio of Long-term debt to common shareholders' equity.
- XOI - It is the NYSE market index for securities in the Oil & Gas sector.

The data is available in this github repository

### Methodology
Securities Classified under the Oil \& Gas sector, which have more than 20 years of Fundamentals data are selected. There are 16 Securities that satisfy this condition. The Fundamentals Data is Quarterly and consists of 80 data points between January 1997 and March 2018. The Stock Price data set, including XOI is selected from 1 Jan 1997 to 1st April 2018.
The following Operations are performed:
- A daily Average price is calculated from the Price dataset from the Open, High, Low and Close Prices. 
- A Monthly Average price is calculated from this daily average price and the Price Dataset, including XOI are merged with the fundamentals dataset.
- A Linear Interpolation is performed on  the Fundamental dataset to obtain monthly data for fundamentals.
- A LSTM model is applied to the dataset. Iterations are performed over the model to choose the most Significant Predictors. This is called the Multivariate Model. Since the Total number of data points for 20 years is only 240 for a particular security, A large number of predictors cannot be selected for the Multivariate model. 
- Tuning and  Optimization of hyperparameters and the number of hidden layers are performed.
- The Results obtained from the Multivariate model are compared with the Bivariate model- Consisting only of the security and Market Index (XOI) and a Univariate model consisting only of the security.

### Running the Model

We need to add all the dataset csv files along with "ML Project_ StockPredictionUsingFundamentals.py" file in one folder. 
Google Collab Platform was used to train and test the model. 

We may directly run "ML Project_ StockPredictionUsingFundamentals.py" in collab or any other platform. 

- Dataset preparation will be done automatically while running the code. 

- The Project has implemented a LSTM Recurrent neural network with three different input sizes. We have termed these models as Univariate, Bivariate and Multivariate models.

For running, each model, we just need to change the dataset input as given below. 

For Univariate Model, use dataset= dd6
For Bivariate Model, use dataset= dd7
For Multivariate Model, use dataset= dd5

- The Code will plot the two graphs for each of the input stocks. First one will be the RMSE plot for training and validation data, while second will show the predicted output results for coming 12 months along with the original stock price. 

### Results

The average % RMSE for the Multivariate case is 54.1%, for the Bivariate case is 57.4% and for the Univariate case is 46.3%. Out of the 16 cases given above, the Multivariate model outperforms the Univariate model 8 out of 16 times, while the Bivariate model gives a better performance than the Univariate model for 8 out of 16 companies. This suggests that overall, the Univariate Model has the best performance. 

It was observed during hyper-parameter tuning that epochs between 20-50 provide the best results on the test dataset. Epochs higher than 50 do not provide sufficient gains in learning to the model. Dropouts were implemented from 0-0.5. The best results were achieved for dropouts ratio ranging from 0-0.1. This may be due to the limited number of data points. An attempt was made to incorporate time distributed output techniques, however, this did not give satisfactory results. This may have been because of the small size of the Test dataset. 



### Future Activities
- [ ] Working on the efficiency of the model for long term stock prediction
- [ ] Projecting the fundamentals separately without the stock prices
- [ ] Designing a portfolio technique to calculate the maximum returns from the predicted fundamentals

