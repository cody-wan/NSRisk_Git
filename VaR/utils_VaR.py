import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from datetime import date
from yahooquery import Ticker
from scipy.stats import ttest_ind


class Security(object):

    df = pd.read_csv('../data/VaR_add_data.csv').astype({'date':'datetime64'}).set_index('date')

    def __init__(self, ticker):
        self._ticker = ticker
        self._t0 = None
        self._t1 = None
        self._df_hist = None
        self._df_pct_change = None
        self._df_risk = None

    def set_df_hist(self, t0, t1, method='yahoo', interval='1d'):
        t0=pd.to_datetime(t0)
        t1=pd.to_datetime(t1)
        self._t0 = t0
        self._t1 = t1

        if self._ticker in ['XAU=', 'CLc1', 'LCOc1', 'GBP=', 'XLK', '.IXTTR', 'XLE', '.IXE', '.RUT']:
            self._df_hist = self.read_from_local(self._ticker)
        else:
            # uses yahooquery.Ticker to read yahoo finance data
            self._df_hist = Ticker(self._ticker).history(start=self._t0, end=self._t1, interval=interval) # indexed from least recent to most


        if interval=='1d':
            self._df_hist.index = self._df_hist.index.normalize() # remove time portion of datetime for daily frequency

    def read_from_local(self, ticker):
        
        return (Security.df[ticker][Security.df[ticker] != 0].dropna()).to_frame()



    
    def get_df_hist(self):
        return self._df_hist

    def set_df_pct_change(self, columns=['adjclose']):
        self._df_pct_change = self._df_hist[columns].pct_change().dropna()
    
    def get_df_pct_change(self):
        return self._df_pct_change

    # compute historical VaR, ES, VaR to ES ratio
    def set_df_risk(self, T, p, rolling_T=None):
        """


        """
        if self._df_pct_change is None:
            raise ValueError('percent change has not been computed')
        
        # initialize new dataframe for storing results
        df = pd.DataFrame(columns=['start', 'end', 'VaR', 'ES', 'returns'])
        # set variable types
        df = df.astype({'VaR': 'float64', 'ES':'float64'})
        # start: starting date, end: ending date of each period we will compute VaR/ES
        df['start'] = self._df_pct_change.iloc[:-T].index
        df['end'] = self._df_pct_change.iloc[T:].index

        if rolling_T is None:
            df = df[::-1][::T][::-1] # compute for each non-overlapping period
        else:
            df = df[::-1][::rolling_T][::-1]

        # compute VaR and ES for each period given by a fixed time window; we move the window forward one unit/day at a time
        for i, row in df.iterrows(): # iterate through each row of df_res
            # subset percent changes in a given period
            returns = self._df_pct_change.loc[row['start']:row['end']].iloc[:-1].values.flatten() # iloc[:-1] -> [start : end), i.e. left close, right open
            df.at[i, 'returns'] = returns
            # computes VaR (p^th-quantile)
            returns_var = np.percentile(returns, q=p)
            df.at[i, 'VaR'] = -returns_var
            # compute ES
            df.at[i, 'ES'] = -np.mean(returns[returns<returns_var])

        self._df_risk = df
    
    def get_df_risk(self):
        return self._df_risk
    
    def label_recession_df(self, peaks, troughs, df='risk'):
        """
            add a column on df that takes value of True(False), if [start, end) period crosses any recession period
        """
        if self._df_risk is None:
            raise ValueError('risk has not been computed')

        # classifies if a period crosses through a recession
        classifier = lambda row, t0, t1 : ((row.start <= t0) & (t0 < row.end)) | ((t0 <= row.start) & (row.start < t1))
        self._df_risk['recession'] = [any([ classifier(row, t0, t1) for t0, t1 in zip(peaks, troughs)]) for row in self._df_risk.itertuples()]
    
    def label_vol_level_df(self, T, df='risk'):
        """
            add a column on df that takes value of ''(''), based on annualized vol level
        """
        if self._df_risk is None:
            raise ValueError('risk has not been computed')

        self._df_risk['vol'] = self._df_risk['returns'].apply(lambda returns: returns.std()*np.sqrt(T))
        vols = self._df_risk['vol'].values
        cut_off = np.median(vols)
        self._df_risk['vol level'] = np.where(self._df_risk['vol'] <= cut_off, 'low', 'high')




# ---------------------- data importing ----------------------



# ---------------------- data processing ----------------------




# ----------------------- visualization -----------------------


def plot_scatter(df, t0, t1, T=1, recession_periods=None):
    """
    shows an interactive scatter plot of VaR, ES, and ratio

    Args:
        df: pandas dataframe, with columns:
            start: pd.datetime, starting date
            end: pd.datetime, ending date
            VaR: numeric, VaR of the period from 'start' to 'end'
            ES: numeric, ES of the period from 'start' to 'end'
        
        t0: 'YYYY-MM-DD', left bound of time series we will use, included
        t1: 'YYYY-MM-DD', right bound of time series we will use, not included
            assuming time is ordered chronologically from left to right, and that t0, t1 as a bound applies only to 'start' date of df

        T: int, we will use a (non-strict) subset by taking one row of df for every 'T'-many rows
                        default is 1, i.e. all rows of df are used

        recession_periods: pandas dataframe, if not provided, no recession periods will be highlighted in the plot,
            if provided, it comes with columns:
            Peak: starting date of a recession period, included
            Trough: ending date of a recession period, not included
            
    returns:
        None

    """
    # convert to datetime
    t0 = pd.to_datetime(t0)
    t1 = pd.to_datetime(t1)
    # select one row of df for 'T' many rows, and that whose start date is older than t0 and younger than t1
    df = df[::-1][::T][::-1]
    df = df[(df['start'] < t1) & (df['start'] >= t0)]

    if isinstance(recession_periods, pd.DataFrame):
        # convert type to datetime if not already, and select values from recession_periods 
        # that would be needed given time horizon of df
        if not all([is_datetime(recession_periods[col]) for col in recession_periods.columns]):
            recession_periods = recession_periods.astype({'Peak':'datetime64', 'Trough':'datetime64'})
        # only uses recession_periods that cross through data's overall time period
        recession_periods = recession_periods.iloc[np.argmin(~(recession_periods['Trough'] > df.iloc[0]['start'])):
                                                            np.argmax(~(recession_periods['Peak'] <= df.iloc[len(df)-1]['end']))] 


        # set up parameters for plotting historical recessions from 1930 to 2009 + covid19 recession
        peaks = recession_periods['Peak'].tolist()
        troughs = recession_periods['Trough'].tolist()
        
        recession_shades = [dict(type="rect",xref="x",yref="paper",
                            x0=t0,y0=0, x1=t1, y1=1,
                            fillcolor="LightSalmon",opacity=0.4,layer="below",line_width=0) for t0, t1 in zip(peaks, troughs)]
        if t1 > pd.to_datetime('2020-2-20'):
            recession_shades.append(dict(type="rect",xref="x",yref="paper",
                                    x0="2020-2-20",y0=0,x1=date.today(),y1=1, 
                                    fillcolor="LightSalmon",opacity=0.4,layer="below",line_width=0))
    else:
        recession_shades = None

    # visualize value of historical realized VaR, ES and their ratio
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(name='VaR', x=df['start'], y=df['VaR'], hovertemplate=' %{y:.2%}'), secondary_y=False)
    fig.add_trace(go.Scatter(name='ES', x=df['start'], y=df['ES'], hovertemplate=' %{y:.2%}'), secondary_y=False)
    fig.add_trace(go.Scatter(name='VaR : ES', x=df['start'], y=df['VaR']/df['ES'], hovertemplate=' %{y:.2f}'), secondary_y=True)
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        hovermode='x unified', 
        autosize=False,
        width=1100,
        height=450,
        margin=dict(l=20,r=20,b=20,t=20,pad=4),
        yaxis=dict(tickformat='.0%', rangemode = 'tozero'),
        xaxis=dict(tickformat='%d %b %Y'),
        # all recession periods will be highlighted
        shapes = recession_shades,
        legend=dict(y=1.12)
    )
    fig.show()


def plot_time_series_histogram(df, t0, t1, T=1, recession_periods=None, date_format='%Y'):
    """
    shows an interactive scatter plot of VaR, ES, and ratio

    Args:
        df: pandas dataframe, with columns:
            start: pd.datetime, starting date
            end: pd.datetime, ending date
            VaR: numeric, VaR of the period from 'start' to 'end'
            ES: numeric, ES of the period from 'start' to 'end'
            returns: list of numeric, list contains values in a given time period, used for making histogram
        
        t0: 'YYYY-MM-DD', left bound of time series we will use, included
        t1: 'YYYY-MM-DD', right bound of time series we will use, not included
            assuming time is ordered chronologically from left to right, and that t0, t1 as a bound applies only to 'start' date of df

        T: int, we will use a (non-strict) subset by taking one row of df for every 'T'-many rows
                usually, T should equal to the time horizon for which VaR, ES are computed for

        recession_periods: pandas dataframe, if not provided, no recession periods will be highlighted in the plot,
            if provided, it comes with columns:
            Peak: starting date of a recession period, included
            Trough: ending date of a recession period, not included
        
        date_format: string, contains datetime format for pd.datetime object, used as text display on 'datetime' axis
            
    returns:
        None

    """
    # convert to datetime
    t0 = pd.to_datetime(t0)
    t1 = pd.to_datetime(t1)
    # select one row of df for 'T'-many rows, and that whose start date is older than t0 and younger than t1
    df = df[::-1][::T][::-1]
    df = df[(df['start'] < t1) & (df['start'] >= t0)]

    recession_flag = False
    if isinstance(recession_periods, pd.DataFrame):
        # convert type to datetime if not already, and select values from recession_periods 
        # that would be needed given time horizon of df
        if not all([is_datetime(recession_periods[col]) for col in recession_periods.columns]):
            recession_periods = recession_periods.astype({'Peak':'datetime64', 'Trough':'datetime64'})
        recession_periods = recession_periods.iloc[np.argmin(~(recession_periods.Trough > df.iloc[0].start)):] 
        # set up parameters for plotting historical recessions from 1930 to 2009 + covid19 recession
        peaks = recession_periods['Peak'].tolist()
        troughs = recession_periods['Trough'].tolist()
        recession_flag = True

    # plot time series of histogram of daily percent change
    fig=go.Figure()
    color = 'blue'
    for row in df.itertuples():
        # generate points that sketch out the contour of a histogram
        a0=np.histogram(row.returns, bins=20, density=False)[0].tolist()
        a0=np.repeat(a0,2).tolist()
        a0.insert(0,0)
        a0.pop()
        a1=np.histogram(row.returns, bins=20, density=False)[1].tolist()
        a1=np.repeat(a1,2)
        if recession_flag:
            # mark periods that go through recession as orange
            if any([((pd.to_datetime(t0) < row.start) & (row.start < pd.to_datetime(t1))) | 
                    ((pd.to_datetime(t0) < row.end) & (row.end < pd.to_datetime(t1))) for t0, t1 in zip(peaks, troughs)]):
                color = 'orange'
            else:
                color = 'blue'
        fig.add_traces(go.Scatter3d(x=[row.Index]*len(a1), y=a1, z=a0, 
                                    mode='lines', line={'color':color},
                                    hovertemplate='%{y:.2%}, %{z}',
                                    name=str(row.start.date())+ ' - ' + str(row.end.date())))
    # add VaR and ES
    fig.add_traces(go.Scatter3d(x=df.index, y=-df['VaR'], z=[0]*len(df), 
                                    mode='lines', line={'color':'purple'},
                                    hovertemplate='%{y:.2%}', name='VaR'
                                    ))
    fig.add_traces(go.Scatter3d(x=df.index, y=-df['ES'], z=[0]*len(df), 
                                    mode='lines', line={'color':'red'},
                                    hovertemplate='%{y:.2%}', name='ES'
                                    ))
    # aspectratio: change the ratio of x,y,z axis's length in display
    fig.update_layout(width=1100,height=500,margin=dict(l=0,r=0,b=20,t=20,pad=4),
                    scene=dict(aspectratio=dict(x=2,y=1, z=0.6), aspectmode = 'manual', 
                    xaxis=dict(tickvals=df.index, ticktext=df['start'].apply(lambda x: '{}'.format(pd.to_datetime(x).strftime(date_format))), title=dict(text='')),
                    yaxis=dict(range=[-0.1,0.1], mirror=True, tickformat='%{.2%}', title=dict(text='daily percentage')),
                    zaxis=dict(showticklabels=False, title=dict(text='frequency')),
                    camera=dict(eye=dict(x=1.25,y=-1.8,z=0.2))))
    fig.show()


if __name__ == '__main__':
    pass