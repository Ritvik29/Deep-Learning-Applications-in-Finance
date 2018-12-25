# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import GRU
#import meanandsd
#import preprocessing 
import os
#import hyperopt
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from datetime import datetime

file_list=['PDS.csv',
 'SBR.csv',
 'RDC.csv',
 'PKD.csv',
 'DO.csv',
 'NE.csv',
 'IO.csv',
 'ESV.csv',
 'NBL.csv',
 'PBT.csv',
 'RRC.csv',
 'SWX.csv',
 'EGN.csv',
 'MUR.csv',
 'OKE.csv',
 'SJT.csv',
 'DVN.csv',
 'NBR.csv',
 'SSL.csv',
 'SWN.csv',
 'HP.csv',
 'E.csv',
 'MTR.csv',
 'MRO.csv',
 'PHX.csv',
 'RIG.csv',
 'EGY.csv',
 'OXY.csv',
 'ESTE.csv',
 'EQT.csv',
 'TOT.csv',
 'SM.csv',
 'NRT.csv',
 'NFX.csv',
 'PES.csv',
 'PXD.csv',
 'EOG.csv',
 'CVX.csv',        
 'XOM.csv']


restricted_list= ['AAPL.csv',
 'AA.csv',
 'ABK.csv',
 'ABMD.csv',
 'ABT.csv',
 'ACE.csv',
 'ACIW.csv',
 'ACXM.csv',
 'ADBE.csv',
 'ADI.csv',
 'ADM.csv',
 'ADP.csv',
 'ADSK.csv',
 'AEE.csv',
 'AEO.csv',
 'AEP.csv',
 'AES.csv',
 'AFG.csv',
 'AFL.csv',
 'AGCO.csv',
 'AGN.csv',
 'AIG.csv',
 'AIV.csv',
 'AJG.csv',
 'AKRX.csv',
 'ALB.csv',
 'ALK.csv',
 'ALL.csv',
 'ALTR.csv',
 'ALXN.csv',
 'AMAT.csv',
 'AMD.csv',
 'AMGN.csv',
 'AMG.csv',
 'AMR.csv',
 'AMSC.csv',
 'AMT.csv',
 'AMZN.csv',
 'ANF.csv',
 'ANSS.csv',
 'AN.csv',
 'AOC.csv',
 'AOS.csv',
 'APA.csv',
 'APC.csv',
 'APD.csv',
 'APH.csv',
 'APOL.csv',
 'APU.csv',
 'ARE.csv',
 'ARG.csv',
 'ARW.csv',
 'ATI.csv',
 'ATVI.csv',
 'AVB.csv',
 'AVP.csv',
 'AVY.csv',
 'AXP.csv',
 'AZO.csv',
 'BAC.csv',
 'BAX.csv',
 'BA.csv',
 'BBBY.csv',
 'BBT.csv',
 'BBY.csv',
 'BCR.csv',
 'BC.csv',
 'BDX.csv',
 'BEN.csv',
 'BF.csv',
 'BHI.csv',
 'BIG.csv',
 'BIIB.csv',
 'BKE.csv',
 'BLL.csv',
 'BMS.csv',
 'BMY.csv',
 'BSX.csv',
 'BWA.csv',
 'BXP.csv',
 'CAG.csv',
 'CAH.csv',
 'CAM.csv',
 'CAT.csv',
 'CA.csv',
 'CBS.csv',
 'CB.csv',
 'CCI.csv',
 'CCL.csv',
 'CDNS.csv',
 'CELG.csv',
 'CERN.csv',
 'CHD.csv',
 'CHK.csv',
 'CHRW.csv',
 'CIEN.csv',
 'CINF.csv',
 'CI.csv',
 'CLF.csv',
 'CLX.csv',
 'CL.csv',
 'CMA.csv',
 'CMI.csv',
 'CMS.csv',
 'COF.csv',
 'COG.csv',
 'COO.csv',
 'COST.csv',
 'CPB.csv',
 'CPN.csv',
 'CRVL.csv',
 'CR.csv',
 'CSCO.csv',
 'CSC.csv',
 'CSX.csv',
 'CTAS.csv',
 'CTL.csv',
 'CTSH.csv',
 'CTXS.csv',
 'CVG.csv',
 'CVS.csv',
 'CVX.csv',
 'CZN.csv',
 'C.csv',
 'DAL.csv',
 'DCI.csv',
 'DDR.csv',
 'DDS.csv',
 'DD.csv',
 'DE.csv',
 'DF.csv',
 'DGX.csv',
 'DHI.csv',
 'DHR.csv',
 'DISH.csv',
 'DIS.csv',
 'DLTR.csv',
 'DNR.csv',
 'DOV.csv',
 'DOW.csv',
 'DO.csv',
 'DRE.csv',
 'DRI.csv',
 'DTE.csv',
 'DVA.csv',
 'DV.csv',
 'D.csv',
 'EBAY.csv',
 'ECL.csv',
 'ED.csv',
 'EFX.csv',
 'EIX.csv',
 'EK.csv',
 'EL.csv',
 'EMC.csv',
 'EMN.csv',
 'EMR.csv',
 'EOG.csv',
 'EQR.csv',
 'EQT.csv',
 'ERTS.csv',
 'ESRX.csv',
 'ESS.csv',
 'ESV.csv',
 'ETFC.csv',
 'ETN.csv',
 'ETR.csv',
 'EXPD.csv',
 'FAST.csv',
 'FCX.csv',
 'FDO.csv',
 'FDS.csv',
 'FDX.csv',
 'FE.csv',
 'FHN.csv',
 'FII.csv',
 'FISV.csv',
 'FITB.csv',
 'FLIR.csv',
 'FLS.csv',
 'FL.csv',
 'FMC.csv',
 'FOSL.csv',
 'FO.csv',
 'FPL.csv',
 'FRT.csv',
 'FRX.csv',
 'F.csv',
 'GAS.csv',
 'GCI.csv',
 'GD.csv',
 'GE.csv',
 'GILD.csv',
 'GIS.csv',
 'GLW.csv',
 'GPC.csv',
 'GPS.csv',
 'GRA.csv',
 'GT.csv',
 'GWW.csv',
 'HAL.csv',
 'HAR.csv',
 'HAS.csv',
 'HBAN.csv',
 'HCN.csv',
 'HCP.csv',
 'HD.csv',
 'HES.csv',
 'HIBB.csv',
 'HIG.csv',
 'HOG.csv',
 'HOLX.csv',
 'HON.csv',
 'HOT.csv',
 'HPQ.csv',
 'HP.csv',
 'HRB.csv',
 'HRL.csv',
 'HRS.csv',
 'HSIC.csv',
 'HSY.csv',
 'HUM.csv',
 'IACI.csv',
 'IBM.csv',
 'IDXX.csv',
 'IFF.csv',
 'IGT.csv',
 'INCY.csv',
 'INTC.csv',
 'INTU.csv',
 'IPG.csv',
 'IP.csv',
 'ITT.csv',
 'ITW.csv',
 'IT.csv',
 'JBHT.csv',
 'JBL.csv',
 'JCI.csv',
 'JDSU.csv',
 'JEC.csv',
 'JNJ.csv',
 'JPM.csv',
 'JWN.csv',
 'KBH.csv',
 'KEY.csv',
 'KIM.csv',
 'KLAC.csv',
 'KMB.csv',
 'KO.csv',
 'KR.csv',
 'KSS.csv',
 'KSU.csv',
 'K.csv',
 'LEG.csv',
 'LH.csv',
 'LIZ.csv',
 'LLTC.csv',
 'LLY.csv',
 'LMT.csv',
 'LM.csv',
 'LNC.csv',
 'LNT.csv',
 'LOW.csv',
 'LRCX.csv',
 'LSI.csv',
 'LSTR.csv',
 'LTD.csv',
 'LTR.csv',
 'LUK.csv',
 'LUV.csv',
 'LXK.csv',
 'MAA.csv',
 'MAC.csv',
 'MAR.csv',
 'MAS.csv',
 'MAT.csv',
 'MBI.csv',
 'MCD.csv',
 'MCHP.csv',
 'MCK.csv',
 'MDP.csv',
 'MDT.csv',
 'MGM.csv',
 'MHK.csv',
 'MHP.csv',
 'MKC.csv',
 'MLM.csv',
 'MMC.csv',
 'MMM.csv',
 'MNST.csv',
 'MNST.csv',
 'MOLX.csv',
 'MOT.csv',
 'MO.csv',
 'MRK.csv',
 'MSFT.csv',
 'MS.csv',
 'MTB.csv',
 'MTD.csv',
 'MTG.csv',
 'MTW.csv',
 'MUR.csv',
 'MU.csv',
 'MYL.csv',
 'M.csv',
 'NAV.csv',
 'NBL.csv',
 'NFX.csv',
 'NKE.csv',
 'NOV.csv',
 'NSC.csv',
 'NTAP.csv',
 'NTRS.csv',
 'NUE.csv',
 'NUS.csv',
 'NU.csv',
 'NWL.csv',
 'NYT.csv',
 'ODP.csv',
 'OI.csv',
 'OKE.csv',
 'OMC.csv',
 'OMX.csv',
 'ORCL.csv',
 'ORLY.csv',
 'OXY.csv',
 'O.csv',
 'PAYX.csv',
 'PBI.csv',
 'PCAR.csv',
 'PCG.csv',
 'PCL.csv',
 'PCP.csv',
 'PDCO.csv',
 'PEG.csv',
 'PEP.csv',
 'PFE.csv',
 'PGR.csv',
 'PG.csv',
 'PHM.csv',
 'PH.csv',
 'PKI.csv',
 'PLD.csv',
 'PLL.csv',
 'PMTC.csv',
 'PNC.csv',
 'PNR.csv',
 'PNW.csv',
 'PPG.csv',
 'PPL.csv',
 'PVH.csv',
 'PWR.csv',
 'PXD.csv',
 'PX.csv',
 'QCOM.csv',
 'QLGC.csv',
 'QSII.csv',
 'RAD.csv',
 'RAVN.csv',
 'RDC.csv',
 'REGN.csv',
 'REG.csv',
 'RHI.csv',
 'RJF.csv',
 'RL.csv',
 'RMD.csv',
 'ROK.csv',
 'ROP.csv',
 'ROST.csv',
 'RRC.csv',
 'RRD.csv',
 'RSG.csv',
 'RSH.csv',
 'RTN.csv',
 'R.csv',
 'SAM.csv',
 'SANM.csv',
 'SBAC.csv',
 'SBUX.csv',
 'SCG.csv',
 'SCHW.csv',
 'SEE.csv',
 'SHW.csv',
 'SIAL.csv',
 'SJM.csv',
 'SLB.csv',
 'SLE.csv',
 'SLG.csv',
 'SLM.csv',
 'SNA.csv',
 'SNDK.csv',
 'SNPS.csv',
 'SO.csv',
 'SPG.csv',
 'SPLS.csv',
 'SRCL.csv',
 'SRE.csv',
 'SSP.csv',
 'STI.csv',
 'STJ.csv',
 'STR.csv',
 'STT.csv',
 'STZ.csv',
 'SVU.csv',
 'SWKS.csv',
 'SWK.csv',
 'SWN.csv',
 'SWY.csv',
 'SYK.csv',
 'SYMC.csv',
 'SYNT.csv',
 'SYY.csv',
 'S.csv',
 'TAP.csv',
 'TEG.csv',
 'TER.csv',
 'TEX.csv',
 'TE.csv',
 'TGT.csv',
 'THC.csv',
 'TIF.csv',
 'TJX.csv',
 'TKR.csv',
 'TMK.csv',
 'TMO.csv',
 'TRV.csv',
 'TSCO.csv',
 'TSN.csv',
 'TSO.csv',
 'TSS.csv',
 'TXN.csv',
 'TXT.csv',
 'TYC.csv',
 'T.csv',
 'UAL.csv',
 'UDR.csv',
 'UHS.csv',
 'UIS.csv',
 'UNH.csv',
 'UNM.csv',
 'UNP.csv',
 'URBN.csv',
 'URI.csv',
 'USB.csv',
 'UTX.csv',
 'VAR.csv',
 'VFC.csv',
 'VIVO.csv',
 'VLO.csv',
 'VNO.csv',
 'VRSN.csv',
 'VRTX.csv',
 'VTR.csv',
 'VZ.csv',
 'WAG.csv',
 'WAT.csv',
 'WDC.csv',
 'WDR.csv',
 'WEC.csv',
 'WEN.csv',
 'WFC.csv',
 'WFMI.csv',
 'WHR.csv',
 'WINA.csv',
 'WMB.csv',
 'WMT.csv',
 'WM.csv',
 'WPI.csv',
 'WPO.csv',
 'WY.csv',
 'XEL.csv',
 'XLNX.csv',
 'XL.csv',
 'XOM.csv',
 'XRAY.csv',
 'XRX.csv',
 'YHOO.csv',
 'YUM.csv',
 'ZION.csv']


 

# Python program to illustrate the intersection 
# of two lists in most simple way 
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3  
Imp_files=intersection(file_list,restricted_list)

columntitles2=['Date','ROA','OHLC_avg',
 'OHLC_avg_xoi', 'Market_Cap',
 'Cumulative dividends per share',
  'Dividend payout ratio',
 'Long-term debt to equity ratio',
 'Net margin',
 'Asset turnover',
 'Free cash flow per share',
 'P/E ratio',
 'P/B ratio',
 'Book value of equity per share']
columntitles=['Date','OHLC_avg',
 'OHLC_avg_xoi','Market_Cap',
 'ROE',
 'ROA',
 'Cumulative dividends per share',
 'Dividend payout ratio',
 'Long-term debt to equity ratio',
 'Current ratio',
 'Net margin',
 'Asset turnover',
 'Free cash flow per share',
 'P/E ratio',
 'P/B ratio',
 'Book value of equity per share']
columntitles2=['Date','ROA','OHLC_avg',
 'OHLC_avg_xoi', 
 'Cumulative dividends per share',
  'Dividend payout ratio',
 'Long-term debt to equity ratio',
 'Net margin',
 'Asset turnover',
 'Free cash flow per share',
 'P/E ratio',
 'P/B ratio',
 'Book value of equity per share']

def prepare_dataset(f1):
  #f1=pd.read_csv('sample_data/'+str(f1),usecols=[0,1,2,3,4])
  fname=f1
  
  
  
  f1=pd.read_csv(str(f1))
  f2=pd.read_csv('XOI.csv')
  f1['Date']=pd.to_datetime(f1['Date'])
  f2['Date']=pd.to_datetime(f2['Date'])
  date_cutoff=datetime(2018,4,1)
  f1=f1[f1['Date']<date_cutoff]
  f2=f2[f2['Date']<date_cutoff]
  
  
  
  ticker=fname.split('.')
  ticker=ticker[0]
  fundamentals=ticker + '_quarterly_financial_data.csv'
  f3=pd.read_csv(fundamentals)
  pd.to_datetime(f1['Date'])

  # TAKING DIFFERENT INDICATORS FOR PREDICTION
  OHLC_avg = 0.25*(f1['Open']+f1['High']+f1['Low']+f1['Close'])
  close_val = f1[['Close']]

  df2=pd.DataFrame()

  df2['Date']=f1['Date']
  df2['OHLC_avg']=OHLC_avg
  df2['close_val']=close_val
  OHLC_Original=OHLC_avg


  pd.to_datetime(f2['Date'])

  # TAKING DIFFERENT INDICATORS FOR PREDICTION-XOI
  OHLC_avg_XOI = 0.25*(f2['Open']+f2['High']+f2['Low']+f2['Close'])
  close_val_XOI = f2[['Close']]

  df3=pd.DataFrame()

  df3['Date']=f2['Date']
  df3['OHLC_avg_XOI']=OHLC_avg_XOI
  df3['close_val_XOI']=close_val_XOI
  OHLC_Original=OHLC_avg_XOI
  df4=pd.DataFrame()
  df4['Date']=df2['Date']
  df4['OHLC_avg']=df2['OHLC_avg']
  df4['OHLC_avg_xoi']=df3['OHLC_avg_XOI']
  df4['Market_Cap']=df2['OHLC_avg']*f1['Volume']
  return df4

def meanandsd2(f1):
  #df1=pd.read_csv('Archive\\'+str(f1) 
  df1=prepare_dataset(f1)
  df1['Date']=pd.to_datetime(df1['Date'])
  df1['mnth_yr'] = df1['Date'].apply(lambda x: x.strftime('%B-%Y'))
  df1['mnth_yr']=pd.to_datetime(df1['mnth_yr'])     
  M1=pd.DataFrame(columns=['means','SD'])
  M1['Means']=df1.groupby('mnth_yr').apply(lambda x:x.OHLC_avg.mean())
  M1['Std']=df1.groupby('mnth_yr').apply(lambda x:x.OHLC_avg.std())
  counter=0
  df2=pd.DataFrame()
  df1['Means']=np.nan
  df1['Std']=np.nan
  for i in M1.index:     
     for j in df1['mnth_yr']:
        if i==j:
          df1.loc[df1['mnth_yr']==i,['Means']]=M1.loc[i,['Means']][0]
          df1.loc[df1['mnth_yr']==j,['Std']]=M1.loc[i,['Std']][0]
          q=df1.groupby('mnth_yr')
          df2=q.mean()
          return (df2)
 # convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

for fname in Imp_files:
  dd2=meanandsd2(fname)
  dd2['Date']=dd2.index
  dd2['mnth_yr2']=dd2['Date'].apply(lambda x: x.strftime('%B-%Y'))
  ticker=fname.split('.')
  ticker=ticker[0]
  fundamentals=ticker + '_quarterly_financial_data.csv'
  fun1=pd.read_csv(fundamentals)
  fun1=fun1.replace('None',np.nan)
  fun1.head()
  fun1['Quarter end']=pd.to_datetime(fun1['Quarter end'])
  fun1.sort_values(by='Quarter end', inplace=True)
  fun2=fun1.loc[fun1['Quarter end'].dt.year>=1997]
  fun2['mnth_yr'] = fun2['Quarter end'].apply(lambda x: x.strftime('%B-%Y'))
  fun2['mnth_yr1']=pd.to_datetime(fun2['mnth_yr'])
  dd3=dd2.merge(fun2[['mnth_yr','ROE','ROA','Cumulative dividends per share','Dividend payout ratio','Long-term debt to equity ratio',
             'Current ratio','Net margin','Asset turnover','Free cash flow per share','P/E ratio',
                    'P/B ratio','Book value of equity per share']],left_on='mnth_yr2',right_on='mnth_yr',how='outer')
  fundamentals=['ROE','ROA','Cumulative dividends per share','Dividend payout ratio','Long-term debt to equity ratio',
             'Current ratio','Net margin','Asset turnover','Free cash flow per share','P/E ratio',
                    'P/B ratio','Book value of equity per share']

  dd4=dd3.loc[:,columntitles2]
  dd4.iloc[:,[i for i in range(1,dd4.shape[1])]]=dd4.iloc[:,[i for i in range(1,dd4.shape[1])]].astype(np.float64)
  columntitles=dd4.columns.tolist()
  dd4=dd4.interpolate(method='linear', axis=0).ffill().bfill()
  dd4=dd4.interpolate(method='linear', axis=0).ffill().bfill()
  dd5=dd4.loc[:,['Date','OHLC_avg',
 'OHLC_avg_xoi',
 'P/E ratio',
 'P/B ratio','Book value of equity per share']]
  dd6=dd4.loc[:,['Date','OHLC_avg']]
  dd7=dd4.loc[:,['Date','OHLC_avg','OHLC_avg_xoi']]
  # load dataset
  dataset = dd7
  #values = dataset.iloc[:,[i for i in range(dataset.shape[1]-1,0,-1)]]
  values=dataset.iloc[:,[i for i in range(1,dataset.shape[1])]]
  # ensure all data is float
  values = values.astype('float32')
  # normalize features
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled = scaler.fit_transform(values)
  # specify the number of lag hours
  n_hours = 12
  n_features = dataset.shape[1]-1
  # frame as supervised learning
  reframed = series_to_supervised(scaled, n_hours, 1)
  # drop columns we don't want to predict

  #print(reframed)
  # split into train and test sets
  values = reframed.values
  n_train_hours = int(values.shape[0]-n_hours)
  train = values[:n_train_hours, :]
  test = values[n_train_hours:, :]
  dataset_original_test=dd5.iloc[n_train_hours:,:]
  dataset_original_train=dd5.iloc[:n_train_hours,:]
  # split into input and outputs
  n_obs = n_hours * n_features
  train_X, train_y = train[:, :n_obs], train[:, -n_features]
  test_X, test_y = test[:, :n_obs], test[:, -n_features]
  print(train_X.shape, len(train_X), train_y.shape)
  # reshape input to be 3D [samples, timesteps, features]
  train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
  test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
  print('train_X',train_X.shape, 'train_y',train_y.shape, 'test_X',test_X.shape, 'test_y',test_y.shape)
  # design network
  model = Sequential()
  #model.add(LSTM(256, input_shape=(train_X.shape[1], train_X.shape[2])))
  #model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True,dropout=0.2))
  model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2]),dropout=0.2))
  model.add(Dense(1))
  model.add(Activation('linear'))
  model.compile(loss='mae', optimizer='adam')
  # fit network
  history = model.fit(train_X, train_y, epochs=50, batch_size=1, validation_data=(test_X, test_y), verbose=0, shuffle=False)
  # plot history
  pyplot.plot(history.history['loss'], label='train')
  pyplot.plot(history.history['val_loss'], label='Validation')
  pyplot.ylabel('Loss- '+str(ticker)+' Stocks')
  pyplot.xlabel('Epochs')
  pyplot.legend()
  pyplot.show()
  print(str(ticker)+' -Validation RMSE',sqrt(sum(np.array(history.history['val_loss'])**2)))
  print(str(ticker)+' -Train RMSE',sqrt(sum(np.array(history.history['loss'])**2)))
  #Test--------------------------------------------------------------------------
  from sklearn.preprocessing import MinMaxScaler
  # make a prediction
  yhat = model.predict(test_X)
  test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
  # invert scaling for forecast
  inv_yhat = np.concatenate((yhat,test_X[:,-(n_features-1):]), axis=1)
  inv_yhat = scaler.inverse_transform(inv_yhat)
  inv_yhat = inv_yhat[:,0]
  # invert scaling for actual
  test_y = test_y.reshape((len(test_y), 1))
  inv_y = np.concatenate((test_y, test_X[:, -(n_features-1):]), axis=1)
  inv_y = scaler.inverse_transform(inv_y)
  inv_y = inv_y[:,0]
  # calculate RMSE
  rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
  percentage_rmse=sqrt((1/inv_y.shape[0])*sum((inv_y-inv_yhat)/inv_y)**2)
  
#----------------------------------------Plotting--------------------------------------------------------------
  print('')
  print('')
  import matplotlib.pyplot as plt
  plt.plot(inv_y, 'g', label = 'original dataset')
  #plt.plot(trainPredictPlot, 'r', label = 'training set ***')
  plt.plot(inv_yhat, 'b', label = 'predicted stock price/test set')
  plt.legend(loc = 'upper right')
  plt.xlabel('Time in Months')
  plt.ylabel('OHLC Value of '+str(ticker)+' Stocks')
  plt.show()
  print(str(ticker),'- Test: percentage_rmse',percentage_rmse)
  print(str(ticker),'-Test RMSE: %.3f' % rmse)

model1 = Sequential()
#model1.add(LSTM(256, input_shape=(train_X.shape[1], train_X.shape[2])))
model1.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
#model1.add(LSTM(100))
model1.add(Dense(1))
model1.add(Activation('linear'))
model1.compile(loss='mae', optimizer='adam')
model1.summary()


