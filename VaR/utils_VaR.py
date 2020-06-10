import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from datetime import date


# ---------------------- visualization utils ----------------------

def plot_scatter(df, t0, t1, period_length=1, recession_periods=None):
    """
    shows an interactive scatter plot of VaR, ES, and ratio

    Args:
        df: pandas dataframe, with columns:
            start: pd.datetime, starting date
            end: pd.datetime, ending date
            VaR: numeric, VaR of the period from 'start' to 'end'
            ES: numeric, ES of the period from 'start' to 'end'
        
        t0: pd.datetime, left bound of time series we will use
        t1: pd.datetime, right bound of time series we will use
            assuming time is ordered chronologically from left to right, and that t0, t1 as a bound applies only to 'start' date of df

        period_length: int, we will use a (non-strict) subset by taking one row of df for every 'period_length'-many rows
                        default is 1, i.e. all rows of df are used

        recession_periods: pandas dataframe, if not provided, no recession periods will be highlighted in the plot,
            if provided, it comes with columns:
            Peak: starting date of a recession period
            Trough: ending date of a recession period
            
    returns:
        None

    """
    # select one row of df for 'period_length' many rows, and that whose start date is older than t0 and younger than t1
    df = df[::-1][::period_length][::-1]
    df = df[(df['start'] < t1) & (df['start'] > t0)]

    if isinstance(recession_periods, pd.DataFrame):
        # convert type to datetime if not already, and select values from recession_periods 
        # that would be needed given time horizon of df
        if not all([is_datetime(recession_periods[col]) for col in recession_periods.columns]):
            recession_periods = recession_periods.astype({'Peak':'datetime64', 'Trough':'datetime64'})
        recession_periods = recession_periods.iloc[np.argmin(~(recession_periods.Trough > df.iloc[0].start)):] 
        # set up parameters for plotting historical recessions from 1930 to 2009 + covid19 recession
        peaks = recession_periods['Peak'].tolist()
        troughs = recession_periods['Trough'].tolist()
        
        recession_shades = [dict(type="rect",xref="x",yref="paper",
                            x0=t0,y0=0, x1=t1, y1=1,
                            fillcolor="LightSalmon",opacity=0.4,layer="below",line_width=0) for t0, t1 in zip(peaks, troughs)]
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
        width=1000,
        height=450,
        margin=dict(l=20,r=20,b=20,t=20,pad=4),
        yaxis=dict(tickformat='.0%', rangemode = 'tozero'),
        xaxis=dict(tickformat='%d %b %Y'),
        # all recession periods will be highlighted
        shapes = recession_shades,
        legend=dict(y=1.12)
    )
    fig.show()


def plot_time_series_histogram(df, t0, t1, period_length=252, recession_periods=None, date_format='%Y'):
    """
    shows an interactive scatter plot of VaR, ES, and ratio

    Args:
        df: pandas dataframe, with columns:
            start: pd.datetime, starting date
            end: pd.datetime, ending date
            VaR: numeric, VaR of the period from 'start' to 'end'
            ES: numeric, ES of the period from 'start' to 'end'
            returns: list of numeric, list contains values in a given time period, used for making histogram
        
        t0: pd.datetime, left bound of time series we will use
        t1: pd.datetime, right bound of time series we will use
            assuming time is ordered chronologically from left to right, and that t0, t1 as a bound applies only to 'start' date of df

        period_length: int, we will use a (non-strict) subset by taking one row of df for every 'period_length'-many rows
                        default is 1, i.e. all rows of df are used

        recession_periods: pandas dataframe, if not provided, no recession periods will be highlighted in the plot,
            if provided, it comes with columns:
            Peak: starting date of a recession period
            Trough: ending date of a recession period
        
        date_format: string, contains datetime format for pd.datetime object, used as text display on 'datetime' axis
            
    returns:
        None

    """
    # select one row of df for 'period_length'-many rows, and that whose start date is older than t0 and younger than t1
    df = df[::-1][::period_length][::-1]
    df = df[(df['start'] < t1) & (df['start'] > t0)]

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