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
# import matplotlib as plt
# import seaborn as sns
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import os
# Page Setup ##################################################################
st.set_page_config(page_title='Ledgr | Forecasting Engine',
                   layout="wide", initial_sidebar_state="expanded")


def inject_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


inject_css()

direc = os.getcwd()

# direc = f'{direc}/Documents/Ledgr'
logofile = f'{direc}/pages/appdata/imgs/Ledgr_Logo_F2.png'

with st.sidebar:
    st.image(logofile, use_container_width=True)
    st.caption("Select a stock, train the algorithm and predict scenarios.")
# Variables & Declarations ###############################################
start_date = dt.datetime(2020, 1, 1)
end_date = dt.datetime.today()
pathtkr = f"{direc}/pages/appdata/tickerlist_y.csv"

tickerdb = pd.read_csv(pathtkr)
tickerlist = tickerdb["SYMBOL"]

# Functions & Cached Resources ######################################


@st.cache_data
def getdata(stock):
    stock = stock + ".NS"

    BSE = yf.Ticker(f'{stock}')
    df = BSE.history(period='max')
    return df

# Pagework 1 - Inputs ###################################################


st.title(":Forecast Engine:")
# Icons and Links ###########################
ytube = f"{direc}/pages/appdata/imgs/ytube.svg"
fbook = f"{direc}/pages/appdata/imgs/fbook.svg"
insta = f"{direc}/pages/appdata/imgs/insta.svg"
linkedin = f"{direc}/pages/appdata/imgs/linkedin.svg"
ledgrblog = f"{direc}/pages/appdata/imgs/Ledgr_Logo_F1.png"
url_ytube = "https://www.youtube.com/@LedgrInc"
url_fb = "https://www.facebook.com/share/1BnXaYvRzV/"
url_insta = 'https://www.instagram.com/alphaledgr/'
url_blog = 'https://www.alphaledgr.com/Blog'
url_linkedin = "https://www.linkedin.com/company/ledgrapp/"
st.write("  ---------------------------------------------------------------  ")

fc1, fc2 = st.columns([2, 3])
with fc1:
    st.caption("Train Ledgr's AI Engines. Forecast Asset Prices.")
    st.info("Chart behaviour, predict price-ranges, observe trajectories.")
with fc2:
    st.video('https://youtu.be/QVGy-AnBR4I?si=Y0gl5QwrR9AoE4ft')

st.write("    -----------------------------------------------------------    ")
with st.form("uinputs"):
    stock = st.selectbox("Please Select a Security Symbol", tickerlist)
    submitted = st.form_submit_button("Proceed")
    if submitted:
        pass


df = getdata(stock)
df.tail(5)
ind = df.index
ind = ind.tz_localize(None)
open = df['Open']
hi = df['High']
lo = df['Low']
close = df['Close']
prof_df_close = pd.DataFrame({"ds": ind, "y": close})
prof_df_close = prof_df_close.set_index(['ds'])
# st.write(prof_df_close)
prof_df_close = prof_df_close.reset_index()
st.write(prof_df_close.tail(5))
# Pagework 2 - Forecasting

m = Prophet()

m.fit(prof_df_close)
future_year = m.make_future_dataframe(periods=365)

forecast_year = m.predict(future_year)
fig1 = m.plot(forecast_year)
#fig2 = m.plot_components(forecast_year)
st.plotly_chart(fig1)
#st.plotly_chart(fig2)
a = plot_plotly(m, forecast_year)
a.update_xaxes(title="Timeline", visible=True, showticklabels=True)
a.update_yaxes(title="Predicted Prices (INR)", visible=True)
a.update_traces(marker_color="green",
                selector=dict(mode='markers'))
b = plot_components_plotly(m, forecast_year)
b.update_xaxes(title="Timeline", visible=True, showticklabels=True)
b.update_yaxes(title="Predicted Prices (INR)", visible=True)
dx = forecast_year.filter(["ds", 'yhat'], axis=1)
dx = dx.set_index(['ds'])
dx.rename(columns={'yhat': 'Predictions'}, inplace=True)
c = px.line(dx)
c.add_trace(go.Scatter(x=dx.index, y=df['Close'], name='Close'))
c.update_xaxes(title='Timeline', showticklabels=True, visible=True)
c.update_yaxes(title="Price Data", visible=True)
c.update_layout(legend=dict(
    orientation="h",
    entrywidth=100,
    yanchor="bottom",
    y=1.02,
    xanchor="right", x=1
))
k1, k2, k3 = st.columns([4, 3, 4])
with k1:
    st.write(" ")
with k2:
    st.subheader("Forecast Plots")
with k3:
    st.write(" ")
st.info(
    """The Real Prices are scattered with the forecast line fit
    over them.""")
st.plotly_chart(a, use_container_width=True)
with st.expander("Get Forecast Data Here!"):
    #       st.write(forecast_year.iloc[-150:])
    st.write(forecast_year.iloc[-100:])
st.info(
    """The Closing Prices and the Predictions are plot identically
    as linear plots for comparison.""")
st.plotly_chart(c, use_container_width=True)

j1, j2, j3 = st.columns([3, 6, 3])
with j1:
    st.write(" ")
with j2:
    st.subheader(f"{stock} Price Trajectory")
with j3:
    st.write(" ")

st.info(
    f"""Track the Price Trajectory of {stock} a particular time
    scale based on historical data, over years,
    months and weeks""")
st.plotly_chart(b, use_container_width=True)

st.write("  ---------------------------------------------------------------  ")

c0, column1, column2, column3, column4, column5, c0a = st.columns(
    [1, 1, 1, 1, 1, 1, 1])
with c0:
    st.write(" ")
with column1:
    st.image(ytube, '[Ledgr\'s YouTube Channel](%s)' % url_ytube, width=60)
with column2:
    st.image(fbook, '[Our Meta Page ](%s)' % url_fb, width=60)
with column3:
    st.image(linkedin,  '[Ledgr @ LinkedIn](%s)' % url_linkedin, width=60)
with column4:
    st.write(" ")
    st.image(ledgrblog,  '[Ledgr\'s Blog ](%s)' % url_blog, width=85)
    st.write(" ")
with column5:
    st.image(insta,  '[Ledgr @ Insta](%s)' % url_insta, width=60)
with c0a:
    st.write(" ")
# # ###################################################################
with st.container():
    f9, f10, f11 = st.columns([2, 5, 1])
    with f9:
        st.write(" ")
    with f10:
        st.write(": 2025 - 2026 | All Rights Reserved  ©  Ledgr Inc.")
        st.write(": alphaLedgr.com | alphaLedgr Technologies Ltd. :")
    with f11:
        st.write(" ")
