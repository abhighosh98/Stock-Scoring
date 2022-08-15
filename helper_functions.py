# Raw Package
import numpy as np
import pandas as pd
#Data Source
import yfinance as yf
import talib
import mplfinance as mpf
import datetime


personal_list = ['BIOCON.NS','COLPAL.NS']

# User Inputs for live data or not
def get_user_inputs():
    live = input('Do you want live data?: yes/ no/ max\n')
    if(live == 'max'):
        g_type = 'line' # Need line graph for more timeline else throws error
    else:
        g_type = 'candle'  # Candlestick plots for daily data
    return(live, g_type)

# functions for resistance and supports in df
def create_top_line(x, plot_sup_res): 
    try:
        return(np.array(plot_sup_res)[np.array(plot_sup_res) > x].min())
    except:
        return(np.nan)

def create_bottom_line(x, plot_sup_res):
    try:
        return(np.array(plot_sup_res)[np.array(plot_sup_res) < x].max())
    except:
        return(np.nan)

    
def pull_price_data(live, stock_ticker, mode = 'normal'):
    if(type(stock_ticker) != str): 
        ticker = list(company_dict.keys())[list(company_dict.values()).index(stock_ticker.value)] # If ticker is not provided in function call then picked from dropdown
        ticker += '.NS'
    else:
        ticker = stock_ticker # passed in function call
    if(live == 'no'):
        if(mode == 'scoring'): # 24 weeks for scoring mode. BUT WHAT IS SCORING MODE
            start = datetime.date.today() - datetime.timedelta(weeks = 24)
            end = datetime.date.today() + datetime.timedelta(days = 1)
            interval = '1d'
        else:
            period = input("Enter period of months") # Period of months for data
            start = datetime.date.today() - datetime.timedelta(weeks = int(period)*4)
            end = datetime.date.today() + datetime.timedelta(days = 1)
            interval = input("Enter interval(1m, 1d, 1wk)")
        data = yf.download(tickers=ticker, start = start, end =end, interval=interval)
    elif(live == 'yes'):            
        data = yf.download(tickers=ticker, period = '2d', interval='1m')
        data.set_index(pd.Series(data.index).apply(lambda x: x.replace(tzinfo=None)), inplace = True)
        data.index.name = 'Date'
    elif(live == 'max'):
        if(mode == 'scoring'):
            interval = '1w'# 1 week interval does not work yet. 
        elif(mode == 'backtest'):
            interval = '1d'
        else:
            interval = input("Enter interval(1d, 1wk)")
        data = yf.download(tickers=ticker, period = 'max', interval=interval)
    return(data)



def plot_graph(data, g_type, plot_sup_res, mode = 'full'):
    if(mode == 'quick'): # Without majority of the indicators
        mpf.plot(data,type = g_type, volume = True,
                hlines=dict(hlines = plot_sup_res, linewidths = 0.5),
                vlines=dict(vlines = [*data['final_score_technical_supres+'].dropna().index.to_list(),
                                    *data['final_score_technical_supres-'].dropna().index.to_list()],
                                    colors= [*['green' for i in range(len(data['final_score_technical_supres+'].dropna().index.to_list()))],
                                             *['black' for i in range(len(data['final_score_technical_supres-'].dropna().index.to_list()))]], 
                                    alpha = 0.3, linewidths= 0.8),
                panel_ratios=(2,1) ,figratio=(2,1), figscale=1)
    else:
        rsi30 = [30] * data.shape[0] # Default lines for RSI at 30 and 70
        rsi70 = [70] * data.shape[0]
         # Choosen Candlestick Patterns from TALib
        bearish_reversal_candlestick = [
            'CDL2CROWS-',
            'CDL3BLACKCROWS-',
            'CDL3INSIDE-',
            'CDL3OUTSIDE-',
            'CDL3LINESTRIKE-',
            'CDLDARKCLOUDCOVER-',
            'CDLEVENINGDOJISTAR-',
            'CDLEVENINGSTAR-',
            'CDLSEPARATINGLINES-',

        ]

        bullish_reversal_candlestick = [
            'CDL3INSIDE+',
            'CDL3STARSINSOUTH+',
            'CDL3OUTSIDE+',
            'CDL3LINESTRIKE+',
            'CDL3WHITESOLDIERS+',
            'CDLHAMMER+',
            'CDLINVERTEDHAMMER+',
            'CDLMORNINGSTAR+',
            'CDLSEPARATINGLINES+',
            'CDLUNIQUE3RIVER+',
            'CDLXSIDEGAP3METHODS+'
        ]

    

        #displaying non na candlesticks 
        candlestick_positive_for_graph = []
        candlestick_negative_for_graph = []

        for candlestick in bearish_reversal_candlestick:
            if(data[candlestick].isna().sum() != data.shape[0]):
                temp = mpf.make_addplot(data[candlestick], type = 'scatter', marker='v', color='red', markersize=50)
                data['final_score_technical_'+candlestick] = temp['data'].apply(lambda x : -1 if(x==x) else x) #final score for candlesticks to sum
                candlestick_negative_for_graph.append(temp)

        for candlestick in bullish_reversal_candlestick:
            if(data[candlestick].isna().sum() != data.shape[0]):
                temp = mpf.make_addplot(data[candlestick], type = 'scatter', marker='^', color='green', markersize=50)
                data['final_score_technical_'+candlestick] = temp['data'].apply(lambda x : 1 if(x==x) else x)  #final score for candlesticks to sum
                candlestick_positive_for_graph.append(temp)
        
         #DOJI signals during a trend
        data['final_score_technical_DOJI-'] = np.where((data[['CDLDOJI','CDLDOJISTAR','CDLDRAGONFLYDOJI']].abs().sum(axis = 1) >0) & (data['current_trend'] > 0 ),-1,np.nan)
        data['final_score_technical_DOJI+'] = np.where((data[['CDLDOJI','CDLDOJISTAR','CDLDRAGONFLYDOJI']].abs().sum(axis = 1) >0) & (data['current_trend'] < 0 ),1,np.nan)
        data['graph_DOJI-'] = data['final_score_technical_DOJI-']
        data['graph_DOJI+'] = data['final_score_technical_DOJI+']
        if(data['final_score_technical_DOJI-'].isna().sum() != data.shape[0]):
            for doji_date in data['final_score_technical_DOJI-'].dropna().index:
                data.loc[doji_date, 'graph_DOJI-'] = data.loc[doji_date]['Close']
        if(data['final_score_technical_DOJI+'].isna().sum() != data.shape[0]):
            for doji_date in data['final_score_technical_DOJI+'].dropna().index:
                data.loc[doji_date, 'graph_DOJI+'] = data.loc[doji_date]['Close']
   
        
        
        if(mode != 'scoring'):
            ap = [
        #RSI
         mpf.make_addplot(data['RSI'], panel =2, secondary_y = False, ylabel='RSI'),
         mpf.make_addplot(rsi30, color='r', panel=2, secondary_y = False),
         mpf.make_addplot(rsi70, color='g', panel=2, secondary_y = False),
        #Bollinger Bands
        mpf.make_addplot(data['Close'], panel = 4, ylabel = 'BBand'),
         mpf.make_addplot(data['upperband'], panel = 4, secondary_y = False),
         mpf.make_addplot(data['middleband'], panel = 4, secondary_y = False),
         mpf.make_addplot(data['lowerband'], panel = 4, secondary_y = False),
        #MACD talib
         mpf.make_addplot(data['macd'], panel = 3, secondary_y = False, ylabel = 'MACD', color = 'red'),
        mpf.make_addplot(data['macdsignal'], panel = 3, ylabel = 'MACD', color = 'black'),
        mpf.make_addplot(data['macdhist'], panel = 3, ylabel = 'MACD', color = 'blue'),
        #DOJI plots
        mpf.make_addplot(data['graph_DOJI-'], type = 'scatter', marker = '+', color = 'red', markersize=50),
        mpf.make_addplot(data['graph_DOJI+'], type = 'scatter', marker = '+', color = 'green', markersize=50),             
    ]


            final_add_plots = [*ap, *candlestick_negative_for_graph, *candlestick_positive_for_graph]
            #print(final_add_plots)
            mpf.plot(data,type=g_type, volume =True, addplot = final_add_plots, 
                     panel_ratios=(3,1) ,figratio=(1,1), figscale=2,
            )
    return(data)



#supporting functions
def isSupport(df,i):
  support = df['Low'][i] < df['Low'][i-1]  and df['Low'][i] < df['Low'][i+1] and df['Low'][i+1] < df['Low'][i+2] and df['Low'][i-1] < df['Low'][i-2]
  return support
def isResistance(df,i):
  resistance = df['High'][i] > df['High'][i-1]  and df['High'][i] > df['High'][i+1] and df['High'][i+1] > df['High'][i+2] and df['High'][i-1] > df['High'][i-2]
  return resistance
def isFarFromLevel(l, levels, s):
   return np.sum([abs(l-x) < s  for x in levels]) == 0

# Finding the signals from indicators
def finding_signals_from_data(data, time_period_multiplier = 1):   
    #adding incremental column for x axis instead of data
    data['y0'] = data.reset_index().index

    #adding candlestick days to df
    for pat in talib.get_function_groups()['Pattern Recognition']:
        candlestick_function = getattr(talib, pat)
        data[pat] = candlestick_function(data['Open'], data['High'], data['Low'], data['Close'])    

    # Marking -220,-100,0,100,200 in correct way for displaying on graph
    for col in talib.get_function_groups()['Pattern Recognition']:
        data[col + '-'] = data.Close[np.where(data[col] < 0 , True, False)]
        data[col + '+'] = data.Close[np.where(data[col] > 0 , True, False)]

    # Separate lists for +ve and -ve candlestick patterns
    candlestick_positive = [c for c in data.columns if (c[-1] == '+')]
    candlestick_negative = [c for c in data.columns if (c[-1] == '-')]

    # EMA   
    data['EMA12' ] = talib.EMA(data["Close"], timeperiod=(12*time_period_multiplier))
    data['EMA26' ] = talib.EMA(data["Close"], timeperiod=(26*time_period_multiplier))

    # Bollinger Bands
    data['upperband'], data['middleband'], data['lowerband'] = talib.BBANDS(data['Close'], timeperiod = (20*time_period_multiplier),
        nbdevup=2,
        nbdevdn=2,
        matype=0)

    #MACD 
    data['macd'], data['macdsignal'], data['macdhist'] = talib.MACD(data['Close'], fastperiod=(12*time_period_multiplier), slowperiod=(26*time_period_multiplier), signalperiod=(9*time_period_multiplier))

    #RSI
    data['RSI'] = talib.RSI(data['Close'],  timeperiod = 14)



    #final score for volume to be shown separately
    volume_thresh = data.Volume.quantile([0.25,0.5,0.7])[0.5]
    data['final_score_technical_Volume'] = data['Volume'].apply(lambda x: 1 if(x > volume_thresh) else np.nan ) # 1 if the volume is more than the 50th quantile implying a spike
    # This score is either positive or negative depending on the overall score. Volume supports the trend. 

    # Final score to be summed for RSI
    data['final_score_technical__RSI+'] = data['RSI'].apply(lambda x: 0.33 if(x<=30) else np.nan) # Attributing lower significance of 0.33 for +ve and -ve signals for RSI. 
    data['final_score_technical__RSI-'] = data['RSI'].apply(lambda x: -0.33 if(x>=70) else np.nan)


    # MACD for graphing vlines
    macd_list_pos = []
    macd_list_neg = []
    # MACD Crossover conditions below
    for i in range(len(data)):
        if((data['macd'].iloc[i] > data['macdsignal'].iloc[i]) and (data['macd'].iloc[i-1] <= data['macdsignal'].iloc[i-1])):
            macd_list_pos.append(data.index[i])
        elif((data['macd'].iloc[i] < data['macdsignal'].iloc[i]) and (data['macd'].iloc[i-1] >= data['macdsignal'].iloc[i-1])):
            macd_list_neg.append(data.index[i])
        else:
            macd_list_pos.append(np.nan)
            macd_list_neg.append(np.nan)


    #Final score for MACD
    data['final_score_technical_MACD+'] = np.nan
    data['final_score_technical_MACD-'] = np.nan
    # Attributing a higher significance to MACD of 2
    for date_macd in pd.Series(macd_list_neg).dropna().values:
        data.loc[date_macd, 'final_score_technical_MACD-'] = -2

    for date_macd in pd.Series(macd_list_pos).dropna().values:
        data.loc[date_macd, 'final_score_technical_MACD+'] = +2

    #Trend from talib
    data['current_trend'] = np.where((data['macd'] > data['macdsignal']),1,-1) # I don't think I use this somewhere. 
    
   


    # Bollinger for graphing vlines
    bband_list_pos = []
    bband_list_neg = []
    # Logic for bollinger band cross overs
    for i in range(len(data)):
        if((data['lowerband'].iloc[i] > data['Close'].iloc[i]) and (data['lowerband'].iloc[i-1] <= data['Close'].iloc[i-1])):
            bband_list_neg.append(data.index[i])
        elif((data['middleband'].iloc[i] > data['Close'].iloc[i]) and (data['middleband'].iloc[i-1] <= data['Close'].iloc[i-1])):
            #(data['Close'].iloc[i] - data['lowerband'].iloc[i])/(data['upperband'].iloc[i] - data['lowerband'].iloc[i]) < 0.05):  #close is within 10% of lower band (close - lower)/(upper - lower) < 0.1
            bband_list_neg.append(data.index[i])
        elif((data['upperband'].iloc[i] < data['Close'].iloc[i]) and (data['upperband'].iloc[i-1] >= data['Close'].iloc[i-1])):
            bband_list_pos.append(data.index[i])
        elif((data['middleband'].iloc[i] < data['Close'].iloc[i]) and (data['middleband'].iloc[i-1] >= data['Close'].iloc[i-1])):
            #(data['Close'].iloc[i] - data['lowerband'].iloc[i])/(data['upperband'].iloc[i] - data['lowerband'].iloc[i]) > 0.95):
            bband_list_pos.append(data.index[i])
        else:
            bband_list_pos.append(np.nan)
            bband_list_neg.append(np.nan)
        
        

    #Final score for Bollinger Bands
    data['final_score_technical_BB+'] = np.nan
    data['final_score_technical_BB-'] = np.nan

    for date_bb in pd.Series(bband_list_neg).dropna().values:
        data.loc[date_bb, 'final_score_technical_BB-'] = -0.5

    for date_bb in pd.Series(bband_list_pos).dropna().values:
        data.loc[date_bb, 'final_score_technical_BB+'] = +0.5

    # Support Resistance Graph
    s =  np.mean(data['High'] - data['Low'])



    levels = []

    
    
    for i in range(2,data.shape[0]-2):
      if isSupport(data,i):
        l = data['Low'][i]
        if isFarFromLevel(l, levels, s):
          levels.append((i,l))
      elif isResistance(data,i):
        l = data['High'][i]
        if isFarFromLevel(l, levels, s):
          levels.append((i,l))

    plot_sup_res = []
    for tup in levels:
        plot_sup_res.append(tup[1])

    # Support Resistance Scores
    data['current_top_line'] = data['Close'].apply(lambda x: create_top_line(x, plot_sup_res))
    data['current_bottom_line'] = data['Close'].apply(lambda x: create_bottom_line(x, plot_sup_res))
    
    #Trend signal support/resistance
    supres_list_pos = []
    supres_list_neg = []
    # Logic for price being above or below the Support Resistance found earlier
    for i in range(len(data)-1):
        if((data['Close'].iloc[i] < data['current_top_line'].iloc[i]) and (data['Close'].iloc[i+1] > data['current_top_line'].iloc[i])):
            supres_list_pos.append(data.index[i+1])
        elif((data['Close'].iloc[i] > data['current_bottom_line'].iloc[i]) and (data['Close'].iloc[i+1] < data['current_bottom_line'].iloc[i])):
            supres_list_neg.append(data.index[i+1])
        else:
            supres_list_pos.append(np.nan)
            supres_list_neg.append(np.nan)

    #Final score for Support/Resistance
    data['final_score_technical_supres+'] = np.nan
    data['final_score_technical_supres-'] = np.nan

    for date_supres in pd.Series(supres_list_neg).dropna().values:
        data.loc[date_supres, 'final_score_technical_supres-'] = -2

    for date_supres in pd.Series(supres_list_pos).dropna().values:
        data.loc[date_supres, 'final_score_technical_supres+'] = +2

    return(data, levels, plot_sup_res)

def is_consolidating(df, percentage=2):
    recent_candlesticks = df[-15:]
    
    max_close = recent_candlesticks['Close'].max()
    min_close = recent_candlesticks['Close'].min()

    threshold = 1 - (percentage / 100)
    if min_close > (max_close * threshold):
        return True        

    return False

def is_breaking_out(df, percentage=2.5):
    last_close = df[-1:]['Close'].values[0]

    if is_consolidating(df[:-1], percentage=percentage):
        recent_closes = df[-16:-1]

        if last_close > recent_closes['Close'].max():
            return True

    return False


#Gives signal of Reversal
def technical_scoring_function(mode = 'all', period = -2, live = 'no', csv_file = 'n200.csv'):
    # Error handling for g_type
    try:
        g_type
    except:
        if(live == 'yes'):
            g_type = 'candle'
        else:
            g_type = 'line'
    # Period means this short reversal signal is calculated over the last 2 days
    tpm = 3 # Set for long term(3+)/ short term trend(1)
    # Dictionary to add company and scores
    pos_score_company = {}
    neg_score_company = {}
    if(mode == 'all'):
        n50 = pd.read_csv(csv_file) # Getting tickers from csv
        nifty50 = []
        for t in n50['Symbol'].values:
            nifty50.append(t+'.NS') # Need to add .NS to get Nifty data
        for t in nifty50:
            ticker = t
            #print(live)
            data = pull_price_data(live, t, mode = 'scoring')
            try:
                if(live == 'no'):
                    data, levels, plot_sup_res = finding_signals_from_data(data, time_period_multiplier = 1)
                else:
                    data, levels, plot_sup_res = finding_signals_from_data(data, time_period_multiplier = tpm)
            except:
                print("Error with", ticker)
                continue

            data = plot_graph(data, g_type, plot_sup_res, mode = 'scoring')


            technical_score_column_list = []
            for col in data.columns:
                if(col.startswith('final_score_technical')): # The columns in the finding_signals_from_data lists the final scores in columns starting with "final_score_technical"
                    technical_score_column_list.append(col)

            
            
            technical_score_df = data[technical_score_column_list].iloc[period:].copy()
            
            # Drop DOJI scores for live only
            if(live=='yes'):
                technical_score_df.drop(columns = ['final_score_technical_DOJI-', 'final_score_technical_DOJI+'], inplace = True)
            # Issue is that when both +ve and -ve scores are there in array, then positive gets triggered and then max of that. 
            # Instead will try to create a flow with no if condition and add or sub volume based on final sum
            if(sum(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values ) > 0):
                #print("Positive Score",ticker)
                if(technical_score_df['final_score_technical_Volume'].isna().sum() != abs(period)):
                    pos_score_company[ticker] = max(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) +1
                else:
                    pos_score_company[ticker] = max(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values)
            elif(sum(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) < 0):
                #print("Negative Score", ticker)
                if(technical_score_df['final_score_technical_Volume'].isna().sum() != abs(period)):
                    pos_score_company[ticker] = min(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) -1
                else:
                    pos_score_company[ticker] = max(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values)
        return(pos_score_company, neg_score_company)

            
    # Specific list
    elif(mode == 'list'):
        nifty50 = personal_list
        for t in nifty50:
            ticker = t
            data = pull_price_data(live, t, mode = 'scoring')
            if(live == 'no'):
                data, levels, plot_sup_res = finding_signals_from_data(data, time_period_multiplier = 1)
            else:
                data, levels, plot_sup_res = finding_signals_from_data(data, time_period_multiplier = tpm)
            data = plot_graph(data, g_type, plot_sup_res, mode = 'scoring')


            technical_score_column_list = []
            for col in data.columns:
                if(col.startswith('final_score_technical')):
                    technical_score_column_list.append(col)

            technical_score_df = data[technical_score_column_list].iloc[period:].copy()
            if(sum(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) > 0):
                #print("Positive Score",ticker)
                if(technical_score_df['final_score_technical_Volume'].isna().sum() != abs(period)):
                    pos_score_company[ticker] = max(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) +1
                else:
                    pos_score_company[ticker] = max(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values)
            elif(sum(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) < 0):
                #print("Negative Score", ticker)
                if(technical_score_df['final_score_technical_Volume'].isna().sum() != abs(period)):
                    pos_score_company[ticker] = min(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) -1
                else:
                    pos_score_company[ticker] = max(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values)
        return(pos_score_company, neg_score_company)
    else:
        ticker = t = mode
        data = pull_price_data(live, t, mode = 'scoring')
        if(live == 'no'):
            data, levels, plot_sup_res = finding_signals_from_data(data, time_period_multiplier = 1)
        else:
            data, levels, plot_sup_res = finding_signals_from_data(data, time_period_multiplier = tpm)
        data = plot_graph(data, g_type, plot_sup_res, mode = 'scoring')


        technical_score_column_list = []
        for col in data.columns:
            if(col.startswith('final_score_technical')):
                technical_score_column_list.append(col)

        technical_score_df = data[technical_score_column_list].iloc[period:].copy()
        if(live=='yes'):
                technical_score_df.drop(columns = ['final_score_technical_DOJI-', 'final_score_technical_DOJI+'], inplace = True)
        if(any(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) > 0):
            #print("Positive Score",ticker)
            
            if(technical_score_df['final_score_technical_Volume'].isna().sum() != abs(period)):
                pos_score_company[ticker] = max(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) +1
            else:
                pos_score_company[ticker] = max(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values)
        elif(any(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) < 0):
            #print("Negative Score", ticker)
            if(technical_score_df['final_score_technical_Volume'].isna().sum() != abs(period)):
                neg_score_company[ticker] = min(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) -1
            else:
                neg_score_company[ticker] = min(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values)
        return(technical_score_df, data)

#Gives long term trend
# Need to remember why I made this function if the period can be set in the function above??
def long_term_trend_scoring_function(mode = 'all', period = -30, live = 'no', csv_file = 'n200.csv'):
    try:
        g_type
    except:
        if(live == 'yes'):
            g_type = 'candle'
        else:
            g_type = 'line'
    # Here the period is for the last 30 days for a longer trend
    tpm = 2
    # Dictionary to add company and scores
    pos_score_company = {}
    neg_score_company = {}
    if(mode == 'all'):
        n50 = pd.read_csv(csv_file)
        nifty50 = []
        for t in n50['Symbol'].values:
            nifty50.append(t+'.NS')
        for t in nifty50:
            ticker = t
            data = pull_price_data(live, t, mode = 'scoring')
            try:
                if(live == 'no'):
                    data, levels, plot_sup_res = finding_signals_from_data(data)
                else:
                    data, levels, plot_sup_res = finding_signals_from_data(data, time_period_multiplier = tpm)
            except:
                print("Error with", ticker)
                continue

            data = plot_graph(data, g_type, plot_sup_res, mode = 'scoring')


            technical_score_column_list = []
            for col in data.columns:
                if(col.startswith('final_score_technical')):
                    technical_score_column_list.append(col)

            technical_score_df = data[technical_score_column_list].iloc[period:].copy()
            if(any(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) > 0):
                #print("Positive Score",ticker)
                if(technical_score_df['final_score_technical_Volume'].isna().sum() != abs(period)):
                    pos_score_company[ticker] = sum(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) +1
            elif(any(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) < 0):
                #print("Negative Score", ticker)
                if(technical_score_df['final_score_technical_Volume'].isna().sum() != abs(period)):
                    neg_score_company[ticker] = sum(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) -1
        return(pos_score_company, neg_score_company)
    elif(mode == 'list'):
        nifty50 = personal_list
        for t in nifty50:
            ticker = t
            data = pull_price_data(live, t, mode = 'scoring')
            if(live == 'no'):
                data, levels, plot_sup_res = finding_signals_from_data(data)
            else:
                data, levels, plot_sup_res = finding_signals_from_data(data, time_period_multiplier = tpm)
            data = plot_graph(data, g_type, plot_sup_res, mode = 'scoring')


            technical_score_column_list = []
            for col in data.columns:
                if(col.startswith('final_score_technical')):
                    technical_score_column_list.append(col)

            technical_score_df = data[technical_score_column_list].iloc[period:].copy()
            if(any(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) > 0):
                #print("Positive Score",ticker)
                if(technical_score_df['final_score_technical_Volume'].isna().sum() != abs(period)):
                    pos_score_company[ticker] = sum(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) +1
            elif(any(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) < 0):
                #print("Negative Score", ticker)
                if(technical_score_df['final_score_technical_Volume'].isna().sum() != abs(period)):
                    neg_score_company[ticker] = sum(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) -1
        return(pos_score_company, neg_score_company)
    else:
        ticker = t = mode
        data = pull_price_data(live, t, mode = 'scoring')
        if(live == 'no'):
            data, levels, plot_sup_res = finding_signals_from_data(data)
        else:
            data, levels, plot_sup_res = finding_signals_from_data(data, time_period_multiplier = tpm)
        data = plot_graph(data, g_type, mode = 'scoring')


        technical_score_column_list = []
        for col in data.columns:
            if(col.startswith('final_score_technical')):
                technical_score_column_list.append(col)

        technical_score_df = data[technical_score_column_list].iloc[period:].copy()
        if(any(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) > 0):
            #print("Positive Score",ticker)
            
            if(technical_score_df['final_score_technical_Volume'].isna().sum() != abs(period)):
                pos_score_company[ticker] = sum(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) +1
        elif(any(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) < 0):
            #print("Negative Score", ticker)
            if(technical_score_df['final_score_technical_Volume'].isna().sum() != abs(period)):
                neg_score_company[ticker] = sum(technical_score_df.drop(columns = ['final_score_technical_Volume']).sum(axis = 1).values) -1
        return(technical_score_df, data)

def calculate_financial_ratios(temp_ticker, quarterly, term = 365):
    try:
        if(quarterly == "no"):
            test_income_statement = yf.Ticker(temp_ticker).financials/10000000
            test_balance_sheet = yf.Ticker(temp_ticker).balance_sheet/10000000
            temp_gen_info = yf.Ticker(temp_ticker).info
        else:
            test_income_statement = yf.Ticker(temp_ticker).quarterly_financials/10000000
            test_balance_sheet = yf.Ticker(temp_ticker).quarterly_balance_sheet/10000000
            temp_gen_info = yf.Ticker(temp_ticker).info
        
        # Current and Previous year financials
        bs_curr_year = test_balance_sheet.iloc[:,0]
        is_curr_year = test_income_statement.iloc[:,0]
        bs_prev_year = test_balance_sheet.iloc[:,1]
        is_prev_year = test_income_statement.iloc[:,1]
    except:
        print(temp_ticker, ' Could not get data')
    
    fr_dict = {}
    
    
    

    try:
        # Balance Sheet
        try:
            current_assets = bs_curr_year['Total Current Assets']
        except:
            current_assets = np.nan
        #current_assets = bs_curr_year['Total Current Assets']

        try:
            current_liabilities = bs_curr_year['Total Current Liabilities']
        except:
            current_liabilities = np.nan  
        #current_liabilities = bs_curr_year['Total Current Liabilities']

        try:
            cash = bs_curr_year['Cash']
        except:
            cash = np.nan 
        #cash = bs_curr_year['Cash']

        try:
            short_term_investments = bs_curr_year['Short Term Investments']
        except:
            short_term_investments = np.nan

        try:
            accounts_receivable = bs_curr_year['Net Receivables']
        except:
            accounts_receivable = np.nan

        try:
            inventory = bs_curr_year['Accounts Payable']
        except:
            inventory = np.nan

        try:
            accounts_payable = bs_curr_year['Accounts Payable']
        except:
            accounts_payable = np.nan
        #accounts_payable = bs_curr_year['Accounts Payable']
        total_equity = (bs_curr_year['Total Stockholder Equity'] + bs_prev_year['Total Stockholder Equity'])/2
        total_assets = bs_curr_year['Total Assets']
        total_liabilities = bs_curr_year['Total Liab']
        non_current_liabilities = total_liabilities - current_liabilities
        ppe = bs_curr_year['Property Plant Equipment']



        # Income Statement
        revenue = is_curr_year['Total Revenue']    
        cogs = is_curr_year['Cost Of Revenue']
        gross_profit = is_curr_year['Gross Profit']
        operating_income = is_curr_year['Operating Income']
        income_before_tax = is_curr_year['Income Before Tax']
        net_income = is_curr_year['Net Income']
        income_tax_expense = is_curr_year['Income Tax Expense']
        ebit = is_curr_year['Ebit']
        interest_expense = abs(is_curr_year['Interest Expense'])
        operating_income = is_curr_year['Operating Income']


        # Liquidity Measurement Ratios
        fr_dict['current_ratio'] = current_assets/current_liabilities # A current ratio of 1.0 or greater is an indication that the company is well-positioned to cover its current or short-term liabilities.
        fr_dict['DSO'] = (accounts_receivable/revenue)*term #DSO tells you how many days after the sale it takes people to pay you on average.
        try:
            fr_dict['DIO'] = (inventory/cogs)*term #DIO tells you how many days inventory sits on the shelf on average.
        except:
            fr_dict['DIO'] = np.nan
        fr_dict['operating_cycle'] = fr_dict['DSO'] + fr_dict['DIO'] # (DSO + DIO )Basically the Operating Cycle tells you how many days it takes for something to go from first being in inventory to receiving the cash after the sale.
        try:
            fr_dict['DPO'] = (accounts_payable/cogs)*term #DPO tells you how many days the company takes to pay its suppliers.
        except:
            fr_dict['DPO'] = np.nan
        fr_dict['CCC'] = fr_dict['operating_cycle'] - fr_dict['DPO'] #The cash conversion cycle (CCC = DSO + DIO – DPO) measures the number of days a company's cash is tied up in the production and sales process of its operations and the benefit it derives from payment terms from its creditors. The shorter this cycle, the more liquid the company's working capital position is. The CCC is also known as the "cash" or "operating" cycle.

        # Profitability Indicator Ratios
        fr_dict['gross_profit_margin'] = gross_profit / revenue # You can think of it as the amount of money from product sales left over after all of the direct costs associated with manufacturing the product have been paid.
        fr_dict['operating_profit_margin'] = operating_income / revenue # If companies can make enough money from their operations to support the business, the company is usually considered more stable.
        fr_dict['pretax_profit_margin'] = income_before_tax / revenue #Profit is the main goal of for-profit organizations. The goal is to make a profit through growth and to grow every year. As a result, one of the most important roles of the financial and investment analyst is to track and forecast profitability.
        fr_dict['net_profit_margin'] = net_income / revenue # Generally, a net profit margin in excess of 10% is considered excellent, though it depends on the industry and the structure of the business.
        fr_dict['effective_tax_rate'] = income_tax_expense / income_before_tax # If there’s one takeaway, it should be that a company’s tax situation is all but a living, breathing organism in its own right.
        fr_dict['return_on_assets'] = net_income / total_assets # ROA Return on assets gives an indication of the capital intensity of the company, which will depend on the industry; companies that require large initial investments will generally have lower return on assets. ROAs over 5% are generally considered good.
        fr_dict['ROCE'] = ebit / (total_assets - current_liabilities) # ROCE shows investors how many dollars in profits each dollar of capital employed generates.

        # Debt Ratios
        fr_dict['debt_ratio'] = total_liabilities / total_assets #T he debt ratio tells us the degree of leverage used by the company.
        fr_dict['interest_coverage_ratio'] = ebit / interest_expense # The lower a company’s interest coverage ratio is, the more its debt expenses burden the company.

        # Operating Performance Ratios
        fr_dict['fixed_asset_turnover'] = revenue / ppe # Calculates how efficiently a company is a producing sales with its machines and equipment.
        fr_dict['asset_turnover'] = revenue / total_assets # The Asset Turnover ratio can often be used as an indicator of the efficiency with which a company is deploying its assets in generating revenue.


        #in-built ratios
        try:
            fr_dict['twoHundredDayAverage'] = temp_gen_info['twoHundredDayAverage']
        except:
            fr_dict['twoHundredDayAverage'] = np.nan
        #fr_dict['twoHundredDayAverage'] = temp_gen_info['twoHundredDayAverage']

        try:
            fr_dict['payoutRatio'] = temp_gen_info['payoutRatio']
        except:
            fr_dict['payoutRatio'] = np.nan
        #fr_dict['payoutRatio'] = temp_gen_info['payoutRatio']

        try:
            fr_dict['fiftyDayAverage'] = temp_gen_info['fiftyDayAverage']
        except:
            fr_dict['fiftyDayAverage'] = np.nan
        #fr_dict['fiftyDayAverage'] = temp_gen_info['fiftyDayAverage']

        try:
            fr_dict['trailingAnnualDividendRate'] = temp_gen_info['trailingAnnualDividendRate']
        except:
            fr_dict['trailingAnnualDividendRate'] = np.nan
        #fr_dict['trailingAnnualDividendRate'] = temp_gen_info['trailingAnnualDividendRate']

        try:
            fr_dict['dividendRate'] = temp_gen_info['dividendRate']
        except:
            fr_dict['dividendRate'] = np.nan
        #fr_dict['dividendRate'] = temp_gen_info['dividendRate']

        try:
            fr_dict['trailing_PE'] = temp_gen_info['trailingPE']
        except:
            fr_dict['trailing_PE'] = np.nan
        #fr_dict['trailing_PE'] = temp_gen_info['trailingPE']

        try:
            fr_dict['market_cap'] = temp_gen_info['marketCap']
        except:
            fr_dict['market_cap'] = np.nan
        #fr_dict['market_cap'] = temp_gen_info['marketCap']

        try:
            fr_dict['priceToSalesTrailing12Months'] = temp_gen_info['priceToSalesTrailing12Months']
        except:
            fr_dict['priceToSalesTrailing12Months'] = np.nan
        #fr_dict['priceToSalesTrailing12Months'] = temp_gen_info['priceToSalesTrailing12Months']

        try:
            fr_dict['forward_PE'] = temp_gen_info['forwardPE']
        except:
            fr_dict['forward_PE'] = np.nan
        #fr_dict['forward_PE'] = temp_gen_info['forwardPE']

        try:
            fr_dict['fiftyTwoWeekHigh'] = temp_gen_info['fiftyTwoWeekHigh']
        except:
            fr_dict['fiftyTwoWeekHigh'] = np.nan
        #fr_dict['fiftyTwoWeekHigh'] = temp_gen_info['fiftyTwoWeekHigh']

        try:
            fr_dict['fiftyTwoWeekLow'] = temp_gen_info['fiftyTwoWeekLow']
        except:
            fr_dict['fiftyTwoWeekLow'] = np.nan
        #fr_dict['fiftyTwoWeekLow'] = temp_gen_info['fiftyTwoWeekLow']

        try:
            fr_dict['enterpriseToRevenue'] = temp_gen_info['enterpriseToRevenue']
        except:
            fr_dict['enterpriseToRevenue'] = np.nan
        #fr_dict['enterpriseToRevenue'] = temp_gen_info['enterpriseToRevenue']

        try:
            fr_dict['profitMargins'] = temp_gen_info['profitMargins']
        except:
            fr_dict['profitMargins'] = np.nan
        #fr_dict['profitMargins'] = temp_gen_info['profitMargins']

        try:
            fr_dict['enterpriseToEbitda'] = temp_gen_info['enterpriseToEbitda']
        except:
            fr_dict['enterpriseToEbitda'] = np.nan
        #fr_dict['enterpriseToEbitda'] = temp_gen_info['enterpriseToEbitda']

        try:
            fr_dict['trailing_EPS'] = temp_gen_info['trailingEps']
        except:
            fr_dict['trailing_EPS'] = np.nan
        fr_dict['forward_EPS'] = temp_gen_info['forwardEps']
        fr_dict['bookValue'] = temp_gen_info['bookValue']
        fr_dict['priceToBook'] = temp_gen_info['priceToBook']
        fr_dict['cmp'] = temp_gen_info['regularMarketPrice']
    except:
        print(temp_ticker," Errored out")
    return(fr_dict)

def create_df_of_financial_ratios(csv_file = 'n200.csv', quarterly = "no"):
    frames = []
    n50 = pd.read_csv(csv_file)
    n50 = n50[['Industry', 'Symbol']]
    n50['Symbol'] = n50['Symbol'].apply(lambda x: x + '.NS')
    i=1
    for sector, t in n50.values:
        print(i, t)
        i = i + 1
        ratio_data_single_ticker = calculate_financial_ratios(t, quarterly = quarterly)
        temp_df = pd.DataFrame(ratio_data_single_ticker, index=[t])
        temp_df['Sector'] = sector
        frames.append(temp_df)
    return(pd.concat(frames))

def check_total_for_getting_sign_of_volume(x):
    if(x < 0):
        return(-1)
    if(x > 0):
        return(1)

def backtest(period = 2, csv_file = 'n100.csv'):
    profit_dict = {} # {Company : Profit }
    if(csv_file == 'NASDAQ 100 Tickers.csv'):
        ticker_list = []
        n_companies = pd.read_csv(csv_file) # Getting tickers from csv
        for t in n_companies['Symbol'].values:
            ticker_list.append(t)
    else:
        ticker_list = []
        n_companies = pd.read_csv(csv_file) # Getting tickers from csv
        for t in n_companies['Symbol'].values:
            ticker_list.append(t+'.NS') # Need to add .NS to get Nifty data
    # print(ticker_list)
    for t in ticker_list:
    # for t in ['ABBOTINDIA.NS', 'ADANIGREEN.NS']:
        # Get Data
        # live, g_type = get_user_inputs()
        data = pull_price_data(live = 'max', stock_ticker = t, mode = 'backtest')
        # print(data.index[-1])
        try:
            data, levels, plot_sup_res = finding_signals_from_data(data, time_period_multiplier = 1)
        except:
            print(t, "Could not get data")
            continue

        data = data[data['Close'].notna()]
        # plot_graph(data, g_type='line', mode = 'quick', plot_sup_res = plot_sup_res)
        data = plot_graph(data, g_type = 'line', plot_sup_res = plot_sup_res, mode = 'scoring')


        # Scores
        technical_score_column_list = []
        for col in data.columns:
            if(col.startswith('final_score_technical')):
                technical_score_column_list.append(col)
        req_col = ['Open', 'High', 'Low', 'Close', 'Adl Close', 'Volume',
                  'current_trend']
        # Fixing volume score according to sum (+ve or -ve)
        data['final_score_technical_Volume'] = data['final_score_technical_Volume'] * data[technical_score_column_list].drop(columns = ['final_score_technical_Volume']).sum(axis = 1).apply(check_total_for_getting_sign_of_volume)
        positive_signal_days = np.where(data[technical_score_column_list].sum(axis = 1) >= 4)
        negative_signal_days = np.where(data[technical_score_column_list].sum(axis = 1) <= -4)
    #     print("Positive Signal\n",data.iloc[positive_signal_days].index, data.iloc[positive_signal_days][technical_score_column_list].sum(axis = 1))
    #     print("\n\nNegative Signal",data.iloc[negative_signal_days].index, data.iloc[negative_signal_days][technical_score_column_list].sum(axis = 1))


        profits = []
        for d in positive_signal_days[0]:
            try:
                profits.append(((data.iloc[d + period]['Close'] - data.iloc[d]['Close']) / data.iloc[d]['Close']) * 100)
            except:
                print("Could not perform", d)
        try:
            profit_dict[t] = sum(profits) / len(profits)
        except:
            print(t, "division error during appending")
            continue
    #     print(profits)
        print(t, "Done")
    return(profit_dict)