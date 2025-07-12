# __author__ = 'R. Sengupta | r_xn'
# __copyright__ = 'Copyright 2023, Ledgr | www.alphaLedgr.com'
# __credits__ = ['r_xn, s.sengupta, adasgupta@gmail.com']
# __license__ = 'Ledgr | alphaledgr.com'
# __version__ = '01.02.04'
# __maintainer__ = 'r_xn@alphaledgr.com'
# __emails__ = 'r_xn@alphaledgr.com / outreach@alphaledgr.com'
# __status__ = 'In active development'

# Imports #####################################################################

import numpy as np
import pandas as pd
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import matplotlib as plt
import seaborn as sns
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import os
# Page Setup ##################################################################
st.set_page_config(page_title='Ledgr | Forecasting Engine', layout="wide", initial_sidebar_state="expanded")
direc = os.getcwd()

# direc = f'{direc}/Documents/Ledgr'
logofile = f'{direc}/pages/appdata/imgs/Ledgr_Logo_F2.png'

with st.sidebar:
    st.image(logofile)
    st.caption("Select a stock, train the algorithm and predict scenarios.")
# Variables & Declarations ####################################################
start_date = dt.datetime(2020, 1, 1)
end_date = dt.datetime.today()
pathtkr = f"{direc}/pages/appdata/tickerlist_y.csv"

tickerdb = pd.read_csv(pathtkr)
tickerlist = tickerdb["SYMBOL"]

# Icons and Links ###########################

ytube = f"{direc}/pages/appdata/imgs/ytube.svg"
fbook = f"{direc}/pages/appdata/imgs/fbook.svg"
insta = f"{direc}/pages/appdata/imgs/insta.svg"
linkedin = f"{direc}/pages/appdata/imgs/linkedin.svg"
ledgrblog = f"{direc}/pages/appdata/imgs/Ledgr_Logo_F2.png"

fc1, fc2 = st.columns(2)
with fc1:
    st.title(":Forecast Engine:")
    st.write("Train Ledgr's AI Engines. Forecast Asset Prices.")
    st.info("Chart behaviour, predict price-ranges, observe trajectories.")
with fc2:
    st.video('https://youtu.be/tqOWAGEnKKQ?si=yW1nz3AVFKvUxGjA')

st.write("    -----------------------------------------------------------    ")

st.subheader("User Inputs")
stock = st.selectbox("Please Select a Security Symbol: ", tickerlist)
ticker = stock + ".NS"


@st.cache_resource
def getdata(ticker):
    ticker = yf.Ticker(stock)
    df = ticker.history(period='max')
    return df


df = getdata(ticker)
df.reset_index([0])
st.write("1", df)
ind = df.index
ind = ind.tz_convert(None)
open = df[['Open']]
hi = df[['High']]
lo = df[['Low']]
close = df[['Close']]
prof_df_close = pd.DataFrame({"ds": ind, "y": close})
prof_df_close.rename(column={"Date", "ds"}, inplace=True)
# prof_df_close = prof_df_close.reset_index()
st.write("2", prof_df_close.tail(5))


# st.stop()
# ##########################################################################
m = Prophet()

m.fit(prof_df_close)
future_year = m.make_future_dataframe(periods=400)
forecast_year = m.predict(future_year)
m.plot(forecast_year)
m.plot_components(forecast_year)
a = plot_plotly(m, forecast_year)
a.update_xaxes(title="Timeline", visible=True, showticklabels=True)
a.update_yaxes(title="Predicted Prices (INR)", visible=True,
               showticklabels=True)
a.update_traces(marker_color="red", selector=dict(mode='markers'))
b = plot_components_plotly(m, forecast_year)
b.update_xaxes(title="Timeline", visible=True, showticklabels=True)
b.update_yaxes(title="Predicted Prices (INR)", visible=True,
               showticklabels=True)
dx = forecast_year.filter(["ds", 'yhat'], axis=1)
dx = dx.set_index(['ds'])
dx.rename(columns={'yhat': 'Predictions'}, inplace=True)
c = px.line(dx)
c.add_trace(go.Scatter(x=dx.index, y=df['Close'], name='Close'))
c.update_xaxes(title='Timeline', showticklabels=True, visible=True)
c.update_yaxes(title="Price Data", showticklabels=True, visible=True)
c.update_layout(legend=dict(
    orientation="h",
    entrywidth=100,
    yanchor="bottom",
    y=1.02,
    xanchor="right", x=1
    ))
st.write("  ---------------------------------------------------------------  ")

k1, k2, k3 = st.columns([4, 3, 4])
with k1:
    st.write(" ")
with k2:
    st.subheader("Forecast Plot")
with k3:
    st.write(" ")

with st.container():
    st.plotly_chart(a, use_container_width=True)
with st.container():
    st.plotly_chart(c, use_container_width=True)

st.write("  ---------------------------------------------------------------  ")
with st.container():
    j1, j2, j3 = st.columns([3, 4, 3])
    with j1:
        st.write(" ")
    with j2:
        st.subheader(f"{stock} Price Trajectory")
    with j3:
        st.write(" ")

st.plotly_chart(b, use_container_width=True)

st.write("  ---------------------------------------------------------------  ")
url_ytube = "https://www.youtube.com/@LedgrInc"
url_fb = "https://www.facebook.com/share/1BnXaYvRzV/"
url_insta = 'https://www.instagram.com/alphaledgr/'
url_blog = 'https://www.alphaledgr.com/Blog'
url_linkedin = "https://www.linkedin.com/company/ledgrapp/"
st.write("  ---------------------------------------------------------------  ")
column1, column2, column3, column4, column5 = st.columns([1, 1, 1, 2, 1])
with column1:
    st.image(ytube, '[Official Channel](%s)' % url_ytube)
with column2:
    st.image(fbook, '[Ledgr\'s FaceBook Page](%s)' % url_fb)
with column3:
    st.image(linkedin,  '[Follow Us!](%s)' % url_linkedin)
with column4:
    st.write(" ")
    st.image(ledgrblog,  '[Ledgr\'s Blog ](%s)' % url_blog)
    st.write(" ")
with column5:
    st.image(insta,  '[Ledgr\'s @ Instagram ](%s)' % url_insta)
# # ###################################################################
with st.container():
    f9, f10, f11 = st.columns([1, 5, 1])
    with f9:
        st.write(" ")
    with f10:
        st.write(": 2025 - 2026 | All Rights Resrved  ©  Ledgr Inc.")
        st.write(": alphaLedgr.com | alphaLedgr Technologies Ltd. :")
    with f11:
        st.write(" ")

