# Created on Mon Dec 12 00:01:11 2022
# !/usr/bin/env python310
# __author__ = 'R. Sengupta | r_xn'
# __copyright__ = 'Copyright 2023, Ledgr | www.alphaLedgr.com'
# __credits__ = ['r_xn@alphaledgr.com, s_sengupta@alphaledgr.com]
# __license__ = 'Ledgr | alphaledgr.com'
# __version__ = '01.02.04'
# __maintainer__ = 'r_xn@alphaledgr.com'
# __emails__ = 'r_xn@alphaledgr.com / response@alphaledgr.com'
# __status__ = 'In active development'

# # Imports and Definitions # #
import pandas as pd
import datetime as dt
import os
import yfinance as yf
# from nsepy import get_history as gh
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
# import seaborn as sns
# import matplotlib as plt
import streamlit as st
from ta.momentum import RSIIndicator
import json
from ta import add_all_ta_features
from ta.utils import dropna
import json
direc = os.getcwd()
st.set_page_config(page_title='Ledgr | Analytics', layout="wide",
                   initial_sidebar_state="expanded")
st.html(
    "<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-7543848340955227"
     crossorigin="anonymous"></script>")
# logofile2 = f"{direc}/pages/appdata/imgs/Ledgr_Logo_F1.png"

pathtkr = f"{direc}/pages/appdata/tickerlist_y.csv"
# Icons and Links ###########################
ytube = f'{direc}/pages/appdata/imgs/ytube.svg'
fbook = f'{direc}/pages/appdata/imgs/fbook.svg'
insta = f'{direc}/pages/appdata/imgs/insta.svg'
linkedin = f'{direc}/pages/appdata/imgs/linkedin.svg'
ledgrblog = f'{direc}/pages/appdata/imgs/Ledgr_Logo_F2.png'
tickerdb = pd.read_csv(pathtkr)
tickerlist = tickerdb["SYMBOL"]
# ################### Sidebar and Inputs###########################
logofile = f'{direc}/pages/appdata/imgs/Ledgr_Logo_F2.png'
# ################################################################
# ###################### #######################################
url_ytube = "https://www.youtube.com/@LedgrInc"
url_fb = "https://www.facebook.com/share/1BnXaYvRzV/"
url_insta = 'https://www.instagram.com/alphaledgr/'
url_blog = 'https://www.alphaledgr.com/Blog'
url_linkedin = "https://www.linkedin.com/company/ledgrapp/"
# ############################################################
with st.sidebar:
    st.image(logofile, use_container_width=True)
    st.caption("Your unified Fintelligence Portal!")
    st.write("Analyze Assets, Get tactical insights!")

mx1, mx2 = st.columns(2)
with mx1:
    st.title(":AnalyticsBox:")
    st.write("**Track Securities.**")
    st.write("**Visualize KPIs, Follow Signals.**")
    st.info("**Perform Technical Analyses, grab insights.**")
with mx2:
    st.video('https://youtu.be/CkMui1TdMqg?si=o1Jq44z8wxWYMsKn')
# Form and Inputs ############################################################
with st.form('inputs'):
    stock = st.selectbox("Please select stock ticker", tickerlist)
    submitted = st.form_submit_button("Proceed")
    if submitted:
        pass

stock2 = stock + ".NS"


@st.cache_resource
def getdata(stock2):
    stock2 = yf.Ticker(stock2)
    df0 = stock2.history(period="5y")
    df0.to_csv(f"{direc}/pages/appdata/OHLC/{stock2}.csv")
    di = df0.index
    df2 = pd.read_csv(f"{direc}/pages/appdata/OHLC/{stock2}.csv", header=[0])
    df2 = df2.set_index(['Date'])
    df2 = df2.drop(['Dividends', 'Stock Splits'], axis=1)
    return df0, di, df2


df0, di, df2 = getdata(stock2)


@st.cache_resource
def dta(df2):
    df = add_all_ta_features(df2,
                             open='Open', high='High', low='Low',
                             close='Close', volume='Volume')
#    df. pd.DataFrame(df2)
    return df


df = dta(df2)
df_col = pd.DataFrame(df.columns)
df.dropna()

# df_col.to_csv("Analytics Columns DF List.csv")
# st.write("DTA", df)


@st.cache_resource
def hi_lo(df):
    df_3H = df['High'].iloc[-3:]
    dh3 = df_3H.max()
    df_3L = df['Low'].iloc[-3:]
    dl3 = df_3L.min()
    df_14H = df['High'].iloc[-14:]
    dh14 = df_14H.max()
    df_14L = df['Low'].iloc[-14:]
    dl14 = df_14L.min()
    dh_52 = df['High'].iloc[-200:]
    dh52 = dh_52.max()
    dl_52 = df['Low'].iloc[-200:]
    dl52 = dl_52.min()
    dh_25 = df['High'].iloc[-100:]
    dh25 = dh_25.max()
    dl_25 = df['Low'].iloc[-100:]
    dl25 = dl_25.min()
    return dh14, dh3, df_14L, df_14H, df_3L, df_3H, dl3, dl14, dh_52, dh52, dl_52, dl52, dh_25, dh25, dl_25, dl25


dh14, dh3, df_14L, df_14H, df_3L, df_3H, dl3, dl14, dh_52, dh52, dl_52, dl52, dh_25, dh25, dl_25, dl25 = hi_lo(
    df)

dh14, dh3, df_14L, df_14H, df_3L, df_3H, dl3, dl14, dh_52, dh52, dl_52, dl52, dh_25, dh25, dl_25, dl25 = hi_lo(
    df)
trendlist = pd.Series(["Moving Averages (MA)", "Moving Average Convergence Divergence (MACD)", "Average Directional Movement Index (ADX)", "Trix (TRIX)", "Mass Index (MI)", "Commodity Channel Index (CCI)", "Detrended Price Oscillator (DPO)", "KST Oscillator (KST)", "Ichimoku Kinkō Hyō (Ichimoku)", "Parabolic Stop And Reverse (Parabolic SAR)", "Schaff Trend Cycle (STC)"])
momenlist = pd.Series(["Relative Strength Index (RSI)", "Stochastic RSI (SRSI)", "True Strength Index (TSI)", "Ultimate Oscillator (UO)", "Stochastic Oscillator (SR)", "Williams %R (WR)", "Awesome Oscillator (AO)", "Kaufmans Adaptive Moving Average (KAMA)", "Rate of Change (ROC)", "Percentage Price Oscillator (PPO)", "Percentage Volume Oscillator (PVO)"])
volatlist = pd.Series(["Average True Range (ATR) & Ulcer Index (UI)",
                       "Bollinger Bands (BB)", "Keltner Channel (KC)",
                       "Donchian Channel (DC)"])
volulist = pd.Series(["Money Flow Index (MFI)", "Accumulation/Distribution Index (ADI)", "On-Balance Volume (OBV)", "Chaikin Money Flow (CMF)", "Force Index (FI)", "Ease of Movement (EoM, EMV)", "Volume-price Trend (VPT)", "Negative Volume Index (NVI)", "Volume Adjusted Average Price (VWAP)"])

figOHLC = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_width=[0.3, 0.7])
figOHLC.add_trace(go.Ohlc(x=df.index, open=df["Open"], high=df["High"],
                          low=df["Low"], close=df["Close"]), row=1, col=1)
figOHLC.update_xaxes(visible=True, showticklabels=True)
figOHLC.add_trace(go.Bar(
    x=df.index, y=df['Volume'], name='Volume Traded', showlegend=False),
    row=2, col=1)
figOHLC.update_layout(xaxis_rangeslider_visible=False, showlegend=False)


@st.cache_resource
def firm_info(stock):
    msft = yf.Ticker(stock+".NS")
    stock_info = msft.info
    df_stock_info = pd.DataFrame(stock_info.items())
    df_stock_info = df_stock_info.set_index([0])
    df_stock_info.rename(columns={1: 'Details'}, inplace=True)
    return df_stock_info


dfi = firm_info(stock)

###############################################################################
st.write("    ----    ")
with st.expander("Expand the Dashboard to visualize Price Data:"):
    st.write(df)

su1, su2, su3 = st.columns([1, 1, 1])
with su1:
    st.subheader("Firm Overview")

    try:
        symbol = dfi.at['symbol', 'Details']
    except Exception:
        st.write('Data Unreported')
    try:
        longName = dfi.at['longName', 'Details']
        st.write(longName)
    except Exception:
        st.write('Data Unreported')
    try:
        st.write(dfi.at['address2', 'Details'])
    except Exception:
        st.write('Data Unreported')
    try:
        industryDisp = dfi.at['industryDisp', 'Details']
        st.write("**Industry** - ", industryDisp)
        sector = dfi.at['sector', 'Details']
        st.write("**Sector** - ", sector)
    except Exception:
        st.write('Data Unreported')
#        fullTimeEmployees = dfi.at['fullTimeEmployees', 'Details']
#        st.write("**Employees** - ", fullTimeEmployees)
with su2:
    st.subheader("Trading Ranges")
    try:
        regularMarketPreviousClose = dfi.at['regularMarketPreviousClose',
                                            'Details']
        st.write("Regular Mkt. Prev. Close - ", regularMarketPreviousClose)
        regularMarketOpen = dfi.at['regularMarketOpen', 'Details']
        st.write("Regular Mkt. Open - ", regularMarketOpen)
        regularMarketDayLow = dfi.at['regularMarketDayLow', 'Details']
        st.write("Regular Mkt. Day. Low - ", regularMarketDayLow)
        regularMarketDayHigh = dfi.at['regularMarketDayHigh', 'Details']
        st.write("Regular Mkt. Day High - ", regularMarketDayHigh)
    except Exception:
        st.write("Data Unreported")

with su3:
    st.subheader("Price Metrics")
    try:
        target_high_price = dfi.at['targetHighPrice', 'Details']
        st.write('Target High - ', target_high_price)
    except Exception:
        st.write('Data Unreported')
    try:
        target_low_price = dfi.at['targetLowPrice', 'Details']
        st.write('Target Low - ', target_high_price)
    except Exception:
        st.write('Data Unreported')
    try:
        target_median_price = dfi.at['targetMedianPrice', 'Details']
        st.write('Target Median - ', target_high_price)
    except Exception:
        st.write('Data Unreported')
    try:
        avg50 = dfi.at['fiftyDayAverage', 'Details']
        st.write('50 Day Avg. - ', avg50, border=True)
        # twoHundredDayAverage = dfi.at['twoHundredDayAverage', 'Details']
        # st.write('200D Avg - ', twoHundredDayAverage)
    except Exception:
        st.write('Data Unreported')
st.write("  --------------  ")

st.subheader("Metrics Dashboard")
bn1, b52, b25, bn2, bn3, bn4 = st.columns([2, 1, 1, 1, 1, 1])
with bn1:
    st.metric("Asset in Focus", f"{stock}")
with b52:
    # st.subheader("52 Week Range")
    st.metric("52-Week-High", round(dh52))
    st.metric("52-Week-Low", round(dl52))
with b25:
    # st.subheader("25 Week Range")
    st.metric("25-Week-High", round(dh25))
    st.metric("25-Week-Low", round(dl25))
with bn2:
    # st.subheader("14 Day Range")
    st.metric("14-Day-High", round(dh14))
    st.metric("14-Day-Low", round(dl14))
with bn3:
    # st.subheader("3 D Range")
    st.metric("3-Day-High", round(dh3))
    st.metric("3-Day-Low", round(dl3))
with bn4:
    st.write("Traded Volume", df["Volume"].iloc[-1])
    st.metric("Latest Closing", df["Close"].iloc[-1].round(2))
with st.container():
    st.subheader(f"Price and Volume Timelines - {stock}")
    st.plotly_chart(figOHLC, use_container_width=True)
    with st.expander("Click here for Raw and Calculated Data"):
        st.write(df.tail(5))

# ############################################################


@st.cache_resource
def sma(df):
    df_sma = df.filter(["Close", "trend_sma_fast", "trend_sma_slow"], axis=1)
    lsma_fast = df_sma["trend_sma_fast"].iloc[-1].round(2)
    lsma_slow = df_sma["trend_sma_slow"].iloc[-1].round(2)

    return df_sma, lsma_fast, lsma_slow


df_sma, lsma_fast, lsma_slow = sma(df)


@st.cache_resource
def ema(df):
    df_ema = df.filter(["Close", "trend_ema_fast", "trend_ema_slow"], axis=1)
    lema_fast = df_ema["trend_ema_fast"].iloc[-1]
    lema_slow = df_ema["trend_ema_slow"].iloc[-1]
    return df_ema, lema_fast, lema_slow


df_ema, lema_fast, lema_slow = ema(df)


@st.cache_resource
def adx(df):
    df_adx = df.filter(["trend_adx"], axis=1)
    df_adx = df_adx.iloc[14:]
    adx_last = df_adx.iloc[-1].values
    df_adx_signal = df.filter(
        ["trend_adx", "trend_adx_pos", "trend_adx_neg"], axis=1)
    df_adx_signal = df_adx_signal.iloc[14:]
    fig_adx = px.line(df_adx)
    fig_adx.update_layout(showlegend=False)
    fig_adx.update_xaxes(visible=True, showticklabels=True)
    fig_adx.update_yaxes(title='ADX', visible=True, showticklabels=True)
    fig_adx_sig = px.line(df_adx_signal)
    fig_adx_sig.update_xaxes(visible=True, showticklabels=True)
    fig_adx_sig.update_yaxes(
        title='ADX Signal', visible=True, showticklabels=True)
    fig_adx_sig.update_layout(showlegend=False)

    return adx_last, fig_adx, fig_adx_sig


adx_last, fig_adx, fig_adx_sig = adx(df)


@st.cache_resource
def trix(df):
    df2 = df.iloc[12:]
    fig_trix = px.bar(df2, x=df2.index, y="trend_trix", color="trend_trix")
    fig_trix.update_layout(showlegend=False)
    fig_trix.update_xaxes(visible=True, showticklabels=True)
    fig_trix.update_yaxes(title='TRIX', visible=True, showticklabels=True)
    l_trix = df["trend_trix"].iloc[-1]
    l_trix2 = df["trend_trix"].iloc[-2]
    slope_trix = l_trix - l_trix2
    # slope_trix2 = l_trix + l_trix2
    slope_trix = slope_trix/l_trix
    return l_trix, fig_trix, slope_trix


l_trix, fig_trix, slope_trix = trix(df)


@st.cache_resource
def mi(df):
    df_mi = df["trend_mass_index"].iloc[31:]
    fig_mi = px.line(df_mi)
    fig_mi.update_layout(showlegend=False)
    fig_mi.update_xaxes(visible=True, showticklabels=True)
    fig_mi.update_yaxes(title='Mass Index', visible=True, showticklabels=True)
    l_mi = df["trend_mass_index"].iloc[-1]
    return l_mi, fig_mi


l_mi, fig_mi = mi(df)


@st.cache_resource
def cci(df):
    fig_cci = px.bar(df, x=df.index, y="trend_cci", color="trend_cci",
                     labels={"trend_cci": 'CCI'})
    fig_cci.update_layout(showlegend=False)
    fig_cci.update_xaxes(visible=True, showticklabels=True)
    fig_cci.update_yaxes(title='Commodity Channel Index',
                         visible=True, showticklabels=True)
    l_cci = df["trend_cci"].iloc[-1]
    return l_cci, fig_cci


l_cci, fig_cci = cci(df)


@st.cache_resource
def dpo(df):
    l_dp = df["trend_dpo"].iloc[-1]
    fig_dpo = px.area(df["trend_dpo"])
    fig_dpo.update_traces(fill='tozeroy', mode='lines')
    fig_dpo.update_layout(showlegend=False)
    fig_dpo.update_xaxes(visible=True, showticklabels=True)
    fig_dpo.update_yaxes(
        title='Detrended Price Oscillator Index', visible=True, showticklabels=True)
    return l_dp, fig_dpo


l_dp, fig_dpo = dpo(df)


@st.cache_resource
def psar(df):
    df_psar_up = df.filter(["trend_psar_up"], axis=1)
    df_psar_ind = df.filter(["trend_psar_up_indicator", "trend_psar_down_indicator"],
                            axis=1)
    fig_psar = go.Figure()
    fig_psar.add_trace(go.Scatter(x=df.index, y=df["Close"],
                        mode='lines',
                        name='lines'))
    fig_psar.add_trace(go.Scatter(x=df.index, y=df["trend_psar_up"],
                        mode='lines+markers',
                        name='lines+markers'))
    fig_psar.add_trace(go.Scatter(x=df.index, y=df["trend_psar_down"],
                        mode='markers', name='markers'))
    l_psar = df_psar_up.iloc[-1]
    l_psar_downi = df['trend_psar_down_indicator'].iloc[-1]
    l_psar_upi = df['trend_psar_up_indicator'].iloc[-1]
    return l_psar, fig_psar, l_psar_downi, l_psar_upi


l_psar, fig_psar, l_psar_downi, l_psar_upi = psar(df)


@st.cache_resource
def rsif(df):
    rsi_21 = RSIIndicator(df['Close'], window=21)
    df["momentum_rsi_21"] = rsi_21.rsi()
    l_rsi21 = df["momentum_rsi_21"].iloc[-1]
    df_rsi = df.filter(["momentum_rsi", "momentum_rsi_21"],
                       axis=1)
    # df_rsi = df_rsi.iloc[8:]

    fig_rsi = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_width=[0.4, 0.6])
    fig_rsi.add_trace(go.Candlestick(x=df.index,
                                     open=df["Open"], high=df["High"],
                                     low=df["Low"], close=df["Close"],
                                     increasing_line_color='cyan',
                                     decreasing_line_color='gray'),
                      row=1, col=1)
    fig_rsi.add_trace(go.Scatter(x=df.index,
                                 y=df['momentum_rsi'], name='RSI 14', showlegend=False),
                      row=2, col=1)
    fig_rsi.add_trace(go.Scatter(x=df.index,
                                 y=df["momentum_rsi_21"], name='RSI 21',
                                 showlegend=False), row=2, col=1)
    fig_rsi.update_layout(xaxis_rangeslider_visible=False)
    fig_rsi.update_layout(showlegend=False)
    fig_rsi_b = px.line(df_rsi)
    fig_rsi_b.add_hrect(y0=50, y1=75, line_width=0,
                        fillcolor="blue", opacity=0.2)
    fig_rsi_b.add_hrect(y0=25, y1=50, line_width=0,
                        fillcolor="yellow", opacity=0.2)
    fig_rsi.update_xaxes(showticklabels=True, visible=True)
    fig_rsi.update_yaxes(title=' ', showticklabels=True, visible=True)
    fig_rsi_b.update_layout(title=f'RSI for {stock}')
    fig_rsi_b.update_layout(showlegend=False)
    l_rsi = df["momentum_rsi"].iloc[-1]

    return fig_rsi, l_rsi, l_rsi21, fig_rsi_b


fig_rsi, l_rsi, l_rsi21, fig_rsi_b = rsif(df)


@st.cache_resource
def srsi(df):
    fig_rsik = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_width=[0.4, 0.6])
    fig_rsik.add_trace(go.Ohlc(x=df.index, open=df["Open"], high=df["High"],
                               low=df["Low"], close=df["Close"]), row=1, col=1)
    fig_rsik.add_trace(go.Scatter(x=df.index, y=df["momentum_stoch_rsi"],
                                  fill='tozeroy', mode='lines',
                                  line_color='blue'), row=2, col=1)
    fig_rsik.update_layout(xaxis_rangeslider_visible=False,
                           height=500, showlegend=False)
    fig_rsik.update_xaxes(visible=True, showticklabels=True)
    fig_rsik.update_yaxes(visible=True, showticklabels=True)

    fig_rsik2 = go.Figure()
    fig_rsik2.add_trace(go.Scatter(x=df.index,
                                   y=df["momentum_stoch_rsi_k"],
                                   name='RSI - K',
                                   fill=None, mode='lines',
                                   line_color='green'))
    fig_rsik2.add_trace(go.Scatter(x=df.index,
                                   y=df["momentum_stoch_rsi_d"],
                                   name='RSI - D', fill='tonexty',
                                   mode='lines', line_color='indigo'))
    fig_rsik2.update_xaxes(visible=True, showticklabels=True)
    fig_rsik2.update_yaxes(visible=True, showticklabels=True)
    fig_rsik2.update_layout(height=500, showlegend=False)
    l_srsi = df["momentum_stoch_rsi"].iloc[-1]
    l_srsi_k = df["momentum_stoch_rsi_k"].iloc[-1]
    l_srsi_d = df["momentum_stoch_rsi_d"].iloc[-1]

    return l_srsi, fig_rsik, l_srsi_k, l_srsi_d, fig_rsik2


l_srsi, fig_rsik, l_srsi_k, l_srsi_d, fig_rsik2 = srsi(df)


@st.cache_resource
def tsi(df):
    df_tsi = df["momentum_tsi"].iloc[15:]
    fig_tsi = go.Figure()
    fig_tsi.add_trace(go.Scatter(x=df.index,
                                 y=df['momentum_tsi'].iloc[14:],
                                 fill='tozeroy'))
    max_tsi = df_tsi.max(axis=0)

    min_tsi = df_tsi.min(axis=0)
    hi_lim_tsi = max_tsi*0.45
    lo_lim_tsi = min_tsi*0.45
    fig_tsi.update_layout(showlegend=False)
    fig_tsi.update_xaxes(title='Timeline', visible=True, showticklabels=True)
    fig_tsi.update_yaxes(title='True Strength Index',
                         visible=True, showticklabels=True)
    l_tsi = df["momentum_tsi"].iloc[-1]
    return fig_tsi, l_tsi, hi_lim_tsi, lo_lim_tsi


fig_tsi, l_tsi, hi_lim_tsi, lo_lim_tsi = tsi(df)


@st.cache_resource
def uo(df):
    df_uo = df["momentum_uo"].iloc[5:]
    fig_uo = px.line(df_uo)
    l_uo = df['momentum_uo'].iloc[-1]
    fig_uo.update_layout(showlegend=False)
    fig_uo.update_xaxes(title='Timeline', visible=True, showticklabels=True)
    fig_uo.update_yaxes(title='Ultimate Oscillator Index',
                        visible=True, showticklabels=True)
    return l_uo, fig_uo


l_uo, fig_uo = uo(df)


@st.cache_resource
def wr(df):
    fig_wr = px.bar(df["momentum_wr"])
    l_wr = df['momentum_wr'].iloc[-1]
    fig_wr.update_layout(showlegend=False)
    fig_wr.update_xaxes(title='Timeline', visible=True, showticklabels=True)
    fig_wr.update_yaxes(title='Williams %R', visible=True, showticklabels=True)
    return l_wr, fig_wr


l_wr, fig_wr = wr(df)


@st.cache_resource
def roc(df):
    fig_roc = px.bar(df["momentum_roc"], color=df['momentum_roc'])
    fig_roc.update_layout(showlegend=False)
    fig_roc.update_xaxes(title='Timeline', visible=True, showticklabels=True)
    fig_roc.update_yaxes(title='Rate of Change', visible=True,
                         showticklabels=True)
    l_roc = df["momentum_roc"].iloc[-1]
    return l_roc, fig_roc


l_roc, fig_roc = roc(df)


@st.cache_resource
def kst(df):
    df_kst = df.filter(["trend_kst", "trend_kst_sig", "trend_kst_diff"],
                       axis=1)
    fig_kst = px.line(df["trend_kst"])
    fig_kst.update_layout(showlegend=False)
    fig_kst.update_xaxes(title='Timeline', visible=True, showticklabels=True)
    fig_kst.update_yaxes(title='Keltner Channels', visible=True,
                         showticklabels=True)
    fig_kst_sig = px.area(df_kst)
    fig_kst_sig.update_layout(showlegend=False)
    fig_kst_sig.update_xaxes(title='Timeline', visible=True,
                             showticklabels=True)
    fig_kst_sig.update_yaxes(title='KST Index, Diff & Signals', visible=True,
                             showticklabels=True)

    l_kst_sig = df['trend_kst_sig'].iloc[-1]
    l_kst = df["trend_kst"].iloc[-1]
    return l_kst, fig_kst, fig_kst_sig, l_kst_sig


l_kst, fig_kst, fig_kst_sig, l_kst_sig = kst(df)


@st.cache_resource
def ichi(df):
    l_ichi = df["trend_ichimoku_conv"].iloc[-1]
    fig_ichi = go.Figure()
    fig_ichi.add_trace(go.Candlestick(x=df.index,
                                      open=df["Open"], high=df["High"],
                                      low=df["Low"], close=df["Close"],
                                      name="OHLC - Price Plot",
                                      increasing_line_color='cyan',
                                      decreasing_line_color='gray'))
    df_ichi2 = df.filter(['trend_ichimoku_conv', "trend_ichimoku_base",
                          "trend_ichimoku_a", "trend_ichimoku_b"])

    fig_ichi.add_trace(go.Scatter(
        x=df.index, y=df['trend_ichimoku_conv'],
        name='Ichimoku Convergence', showlegend=True))
    fig_ichi.add_trace(go.Scatter(
        x=df.index, y=df["trend_ichimoku_base"],
        name='Ichimoku Base', showlegend=True))
#  fig_ichi.add_trace(go.Scatter(
#      x=df.index, y=df["trend_ichimoku_a"], name='Ichimoku A',
#        showlegend=False))
#  fig_ichi.add_trace(go.Scatter(
#       x=df.index, y=df["trend_ichimoku_b"], name='Ichimoku B',
#  showlegend=False))
    l_ichi_b = df["trend_ichimoku_base"].iloc[-1]

    fig_ichi.update_layout(xaxis_rangeslider_visible=False)
    fig_ichi.update_layout(showlegend=False)
    fig_ichi.update_xaxes(visible=True, showticklabels=True)
    fig_ichi.update_yaxes(title='Ichimoku Indices', visible=True,
                          showticklabels=True)
    fig_ichi3 = px.line(df_ichi2)
    fig_ichi2 = go.Figure()
    fig_ichi2.add_trace(go.Scatter(
        x=df.index, y=df["trend_visual_ichimoku_a"],
        name='Ichimoku A - Visual', showlegend=False))
    fig_ichi2.add_trace(go.Scatter(
        x=df.index, y=df["trend_visual_ichimoku_b"],
        name='Ichimoku B - Visual', showlegend=False))
    # fig_ichi2.update_layout(xaxis_rangeslider_visible=False)
    fig_ichi2.update_layout(height=350, showlegend=False)
    fig_ichi2.update_xaxes(visible=True, showticklabels=True)
    fig_ichi2.update_yaxes(title='Ichimoku Visual', visible=True,
                           showticklabels=True)
    fig_ichi3.update_layout(height=350, showlegend=False)
    fig_ichi3.update_xaxes(visible=True, showticklabels=True)
    fig_ichi3.update_yaxes(title='Ichimoku', visible=True, showticklabels=True)
    return fig_ichi, l_ichi, fig_ichi2, fig_ichi3, l_ichi_b


fig_ichi, l_ichi, fig_ichi2, fig_ichi3, l_ichi_b = ichi(df)


@st.cache_resource
def stc(df):
    fig_stc = px.line(df["trend_stc"])
    fig_stc.update_layout(showlegend=False)
    fig_stc.update_xaxes(title='Timeline', visible=True, showticklabels=True)
    fig_stc.update_yaxes(title='STCI', visible=True, showticklabels=True)
    l_stc = df['trend_stc'].iloc[-1]
    return l_stc, fig_stc


l_stc, fig_stc = stc(df)


@st.cache_resource
def stoch(df):
    df_stoch = df.filter(["momentum_stoch", "momentum_stoch_signal"], axis=1)
    fig_stoch = px.area(df_stoch)
    l_stoch = df["momentum_stoch"].iloc[-1]
    fig_stoch.update_layout(showlegend=False)
    fig_stoch.update_xaxes(title='Timeline', visible=True, showticklabels=True)
    fig_stoch.update_yaxes(title='Stochastic Oscillator Index',
                           visible=True, showticklabels=True)
    fig_stoch_sig = px.bar(df["momentum_stoch_signal"])
    fig_stoch_sig.update_layout(showlegend=False)
    fig_stoch_sig.update_xaxes(
        title='Timeline', visible=True, showticklabels=True)
    fig_stoch_sig.update_yaxes(
        title='Stoch Signal', visible=True, showticklabels=True)
    return fig_stoch, fig_stoch_sig, l_stoch


fig_stoch, fig_stoch_sig, l_stoch = stoch(df)


@st.cache_resource
def ao(df):
    fig_ao = px.histogram(df["momentum_ao"], x=df.index, )
    fig_ao.update_layout(showlegend=False)
    fig_ao.update_xaxes(title='Timeline', visible=True, showticklabels=True)
    fig_ao.update_yaxes(title='Awesome Oscillator',
                        visible=True, showticklabels=True)
    l_ao = df["momentum_ao"].iloc[-1]
    return l_ao, fig_ao


l_ao, fig_ao = ao(df)


@st.cache_resource
def kama(df):
    fig_kama = px.area(df["momentum_kama"])
    fig_kama.update_layout(showlegend=False)
    fig_kama.update_xaxes(title='Timeline', visible=True, showticklabels=True)
    fig_kama.update_yaxes(title='Kaufmanns Moving Average',
                          visible=True, showticklabels=True)
    l_kama = df["momentum_kama"].iloc[-1]
    return fig_kama, l_kama


fig_kama, l_kama = kama(df)


@st.cache_resource
def ppo(df):
    fig_ppo = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.04, row_width=[0.4, 0.7])
    fig_ppo.add_trace(go.Candlestick(x=df.index,
                                     open=df["Open"], high=df["High"],
                                     low=df["Low"], close=df["Close"],
                                     name="Price against Moving Averages",
                                     increasing_line_color='cyan',
                                     decreasing_line_color='gray'), row=1, col=1)
    fig_ppo.add_trace(go.Scatter(
        x=df.index, y=df["momentum_ppo"], name='PPO', showlegend=False), row=2, col=1)
    fig_ppo.add_trace(go.Scatter(
        x=df.index, y=df["momentum_ppo_hist"], name='PPO Histogram', showlegend=False), row=2, col=1)
    fig_ppo.add_trace(go.Scatter(
        x=df.index, y=df["momentum_ppo_signal"], name='PPO Signal', showlegend=False), row=2, col=1)
    fig_ppo.update_layout(xaxis_rangeslider_visible=False, showlegend=False)

    fig_ppo.update_xaxes(title='Timeline', visible=False, showticklabels=True)
    fig_ppo.update_yaxes(title=' ',
                         visible=True, showticklabels=True)
    fig_ppo_signal = px.bar(df["momentum_ppo_signal"])
    fig_ppo_signal.update_layout(title=f'PPO Signal for {stock}',
                                 showlegend=False)

    return fig_ppo, fig_ppo_signal


fig_ppo, fig_ppo_signal = ppo(df)


@st.cache_resource
def pvo(df):
    fig_pvo = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.02, row_width=[0.5, 0.5])
    fig_pvo.add_trace(go.Candlestick(x=df.index,
                                     open=df["Open"], high=df["High"],
                                     low=df["Low"], close=df["Close"],
                                     name="Price against Moving Averages",
                                     increasing_line_color='cyan', decreasing_line_color='gray'),
    row=1, col=1)
    fig_pvo.add_trace(go.Scatter(
        x=df.index, y=df["momentum_pvo"], name='PVO', showlegend=False),
    row=2, col=1)
    fig_pvo.add_trace(go.Scatter(
        x=df.index, y=df["momentum_pvo_hist"], name='PVO Histogram', showlegend=False),
    row=2, col=1)
    fig_pvo.add_trace(go.Scatter(
        x=df.index, y=df["momentum_pvo_signal"],
        name='PVO Signal',
        showlegend=False),
        row=2, col=1)
    fig_pvo.update_layout(xaxis_rangeslider_visible=False, showlegend=False)

    fig_pvo.update_xaxes(title='Timeline', visible=False, showticklabels=True)
    fig_pvo.update_yaxes(title=' ', visible=True, showticklabels=True)
    fig_pvo_signal = px.bar(df["momentum_pvo_signal"])
    fig_pvo_signal.update_layout(
        title=f'PVO Signal for {stock}', showlegend=False)

    return fig_pvo, fig_pvo_signal


fig_pvo, fig_pvo_signal = pvo(df)


@st.cache_resource
def atr(df):
    df_atr = df["volatility_atr"]
    fig_atr = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig_atr.add_trace(go.Candlestick(x=df.index,
                                     open=df["Open"], high=df["High"],
                                     low=df["Low"], close=df["Close"],
                                     name="OHLC - Price Plot",
                                     increasing_line_color='cyan',
                                     decreasing_line_color='gray'),
                      row=1, col=1)
    fig_atr.add_trace(go.Scatter(x=df.index, y=df_atr, name='Average True Range',
                                 showlegend=False), row=2, col=1)
    fig_atr.update_layout(xaxis_rangeslider_visible=False)
    fig_atr.update_layout(title=f'Average true range for {stock}')
    fig_atr.update_layout(showlegend=False)
    fig_atr.update_xaxes(title='Timeline', visible=True, showticklabels=True)
    fig_ui = px.bar(df['volatility_ui'])
    fig_ui.update_xaxes(
        title='Timeline', visible=True, showticklabels=True)
    fig_ui.update_yaxes(title='ATR & Ulcer Index',
                            visible=True, showticklabels=True)
    fig_ui.update_layout(height=400, showlegend=False)
    l_ui = df['volatility_ui'].iloc[-1]
    l_atr = df["volatility_atr"].iloc[-1]
    return l_atr, fig_atr, l_ui, fig_ui


l_atr, fig_atr, l_ui, fig_ui = atr(df)


@st.cache_resource
def dr(df):
    l_dr = df["others_dr"].iloc[-1]
    fig_dr = go.Figure()
    fig_dr.add_trace(go.Scatter(
        x=df.index, y=df['others_dr'], name='Direct Returns'))
    fig_dr.add_trace(go.Scatter(
        x=df.index, y=df['others_dlr'], name='Logarithmic Returns'))
    fig_dr.add_trace(go.Scatter(
        x=df.index, y=df['others_cr'], name='Cumulative Returns'))
    l_dlr = df["others_dlr"].iloc[-1]
    delta_r = l_dr - l_dlr
    fig_dlr = px.line(df["others_dlr"].iloc[10:])
    fig_dlr.update_layout(height=400, title=f"Direct Logarithmic Returns for {stock}",
                          showlegend=False)
    fig_dr.update_layout(height=400, title=f"Direct and Log Returns for {stock}",
                         showlegend=False)
    return l_dr, fig_dr, l_dlr, delta_r, fig_dlr


l_dr, fig_dr, l_dlr, delta_r, fig_dlr = dr(df)


@st.cache_resource
def cr(df):
    fig_cr = px.line(df["others_cr"])
    l_cr = df["others_cr"].iloc[-1]
    fig_cr.update_layout(height=400,
                         title=f"Cumulative Returns for {stock}", showlegend=False)
    fig_cr.update_xaxes(title='Timeline', visible=True, showticklabels=True)
    fig_cr.update_yaxes(title='Cumulative Returns',
                        visible=True, showticklabels=True)
    return l_cr, fig_cr


l_cr, fig_cr = cr(df)


@st.cache_resource
def mfi(df):
    fig_mfi = px.line(df["volume_mfi"].iloc[10:])
    l_mfi = df['volume_mfi'].iloc[-1]
    fig_mfi.update_layout(height=400,
                          title=f'Money Flow for {stock}', showlegend=False)
    fig_mfi.update_xaxes(title='Timeline', showticklabels=True, visible=True)
    fig_mfi.update_yaxes(title="Money Flow", showticklabels=True, visible=True)
    return l_mfi, fig_mfi


l_mfi, fig_mfi = mfi(df)


@st.cache_resource
def adi(df):
    fig_adi = px.area(df["volume_adi"])
    fig_adi.update_layout(title=f'Calculated ADI for {stock}',
                          showlegend=False)
    fig_adi.update_xaxes(title='Timeline', showticklabels=True, visible=True)
    fig_adi.update_yaxes(
        title=f'ADI for {stock}', showticklabels=True, visible=True)
    l_adi = df["volume_adi"].iloc[-1]
    l_adi = l_adi/100000
    return l_adi, fig_adi


l_adi, fig_adi = adi(df)


@st.cache_resource
def obvf(df):
    fig_obv = px.line(df["volume_obv"])
    fig_obv.update_layout(title=f'OBV Signal for {stock}', showlegend=False)
    fig_obv.update_xaxes(title='Timeline', showticklabels=True, visible=True)
    fig_obv.update_yaxes(title="On-Balance Volume",
                         showticklabels=True, visible=True)
    # fig_obv.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig_obv.update_layout(height=400)
# , uniformtext_minsize=8, uniformtext_mode='hide')
    l_obv = df["volume_obv"].iloc[-1]
    l_obv = l_obv/100000
    return l_obv, fig_obv


l_obv, fig_obv = obvf(df)


@st.cache_resource
def cmf(df):
    df_cmf = df["volume_cmf"].iloc[4:]
    fig_cmf = px.bar(df, x=df.index, y="volume_cmf", color="volume_cmf",
                     labels={"volume_cmf": 'CMFI'})
    l_cmf = df["volume_cmf"].iloc[-1]
    fig_cmf.update_layout(
        title=f'Chaikins Money Flow Index for {stock}', showlegend=False)
    fig_cmf.update_xaxes(title='Timeline', showticklabels=True, visible=True)
    fig_cmf.update_yaxes(title="Chaikins Money Flow",
                         showticklabels=True, visible=True)
    return l_cmf, fig_cmf


l_cmf, fig_cmf = cmf(df)


@st.cache_resource
def fi(df):
    fig_fi = px.area(df['volume_fi'])
    l_fi = df["volume_fi"].iloc[-1]
    fig_fi.update_layout(
        title=f'Force Index for {stock}')
    fig_fi.update_layout(showlegend=False)
    fig_fi.update_xaxes(title='Timeline', showticklabels=True, visible=True)
    fig_fi.update_yaxes(title="Force Index", showticklabels=True, visible=True)
    return l_fi, fig_fi


l_fi, fig_fi = fi(df)


@st.cache_resource
def em(df):
    df_em = df.filter(["volume_em", "volume_sma_em"], axis=1)
    fig_em = px.area(df_em)
    fig_em.update_layout(title='Ease of Movement', showlegend=False)
    fig_em.update_yaxes(title='Ease of Movement Index',
                        showticklabels=True, visible=True)
    fig_em.update_xaxes(title='Ease of Movement Index',
                        showticklabels=True, visible=False)

    fig_sma_em = px.line(df["volume_sma_em"])
    fig_sma_em.update_layout(title='Ease of Movement - SMA', showlegend=False)
    l_em = df["volume_em"].iloc[-1]

    l_sma_em = df["volume_sma_em"].iloc[-1]
    l_sma_em2 = df["volume_sma_em"].iloc[-2]

    delta_em = l_em - l_sma_em
    return fig_em, fig_sma_em, l_em, l_sma_em, delta_em


fig_em, fig_sma_em, l_em, l_sma_em, delta_em = em(df)


@st.cache_resource
def nvi(df):
    fig_nvi = px.line(df["volume_nvi"])
    fig_nvi.update_layout(showlegend=False)
    fig_nvi.update_xaxes(title='Timeline', showticklabels=True, visible=True)
    fig_nvi.update_yaxes(title="Negative Volume Index",
                         showticklabels=True, visible=True)
    l_nvi = df["volume_nvi"].iloc[-1]
    return l_nvi, fig_nvi


l_nvi, fig_nvi = nvi(df)


@st.cache_resource
def vwap(df):
    fig_vwap = px.line(df["volume_vwap"])
    fig_vwap.update_layout(showlegend=False)
    fig_vwap.update_xaxes(title='Timeline', showticklabels=True, visible=True)
    fig_vwap.update_yaxes(title="Volume Weighted Average Price",
                          showticklabels=True, visible=True)
    l_vw = df["volume_vwap"].iloc[-1]
    return l_vw, fig_vwap


l_vw, fig_vwap = vwap(df)


@st.cache_resource
def vpt(df):
    fig_vpt = px.bar(df["volume_vpt"])
    fig_vpt.update_layout(showlegend=False)
    fig_vpt.update_xaxes(title='Timeline', showticklabels=True, visible=True)
    fig_vpt.update_yaxes(title="Volume Price Trend Index",
                         showticklabels=True, visible=True)
    l_vp = df["volume_vpt"].iloc[-1]
    return l_vp, fig_vpt


l_vp, fig_vpt = vpt(df)


@st.cache_resource
def bb(df):
    df_bb1 = df.filter(
        ["Close", "volatility_bbm", "volatility_bbh", "volatility_bbl"], axis=1)
    df_bb2 = df.filter(["volatility_bbw"], axis=1)
    df_bb_ind = df.filter(["volatility_bbhi", "volatility_bbli"], axis=1)
    l_bbp = df["volatility_bbp"].iloc[-1]
    fig_bb = px.line(df_bb1)
    fig_bb.update_layout(showlegend=False)
    fig_bb.update_xaxes(title='Timeline', showticklabels=True, visible=True)
    fig_bb.update_yaxes(title="Bollingers Bands",
                        showticklabels=True, visible=True)
    l_bbw = df["volatility_bbw"].iloc[-1]
    fig_bb2 = make_subplots(rows=2, cols=1, shared_xaxes=False,
                            subplot_titles=(
                                "Percentile", "Bandwidth / Squeeze [%]"),
                            row_width=[0.35, 0.65])
    fig_bb2.add_trace(go.Scatter(
        x=df.index, y=df["volatility_bbp"], name='Percentile', showlegend=False), row=1, col=1)
    fig_bb2.update_xaxes(visible=True, showticklabels=True)
    fig_bb2.add_trace(go.Scatter(
        x=df.index, y=df['volatility_bbw'], name='"Bandwidth / Squeeze [%]', showlegend=False), row=2, col=1)
    # fig_bb3.add_trace(go.Bar(x=df.index, y = df_bb_ind))
    # fig_bb2.update_layout(height = 400, showlegend=False)
    # fig_bb2.update_xaxes(title='Timeline', showticklabels=True, visible=True)
    # fig_bb2.update_yaxes(title="x, showticklabels=True, visible=True)
    # fig_bb3.update_layout(showlegend=False)
    # fig_bb3.update_xaxes(title='Timeline', showticklabels=True, visible=True)
    # fig_bb3.update_yaxes(title="BB Percentile [%]", showticklabels=True, visible=True)
    # fig_bb_ind = px.bar(df_bb_ind)
    # fig_bb_ind.update_layout(height = 400, showlegend=False)
    # fig_bb_ind.update_xaxes(title='Timeline', showticklabels=True, visible=True)
    # fig_bb_ind.update_yaxes(title="BB Indicator", showticklabels=True, visible=True)
    l_bbhi = df["volatility_bbhi"].iloc[-1]
    l_bbli = df['volatility_bbli'].iloc[-1]
    return df_bb1, df_bb2, df_bb_ind, l_bbp, fig_bb, fig_bb2, l_bbhi, l_bbli, l_bbw


df_bb1, df_bb2, df_bb_ind, l_bbp, fig_bb, fig_bb2, l_bbhi, l_bbli, l_bbw = bb(
    df)


@st.cache_resource
def donc(df):
    fig_dcw = px.line(df["volatility_dcw"])
    fig_dcw.update_yaxes(title='Donchian Squeeze or Bandwidth',
                         showticklabels=True, visible=True)
    fig_dcw.update_xaxes(title=' ', showticklabels=True, visible=True)
    fig_dcp = px.line(df["volatility_dcp"])
    fig_dcp.update_yaxes(title='Donchian Channel Percentile',
                         showticklabels=True, visible=True)
    fig_dcp.update_xaxes(title=' ', showticklabels=True, visible=True)
    df_dc = df.filter(
        ["volatility_dcl", "volatility_dch", "volatility_dcm"], axis=1)
    fig_dc = px.line(df_dc)
    fig_dc.update_yaxes(
        title=f'Donchian Channels for {stock}',
        showticklabels=True, visible=True)
    fig_dc.update_xaxes(title=' ', showticklabels=True, visible=True)
    fig_dc.update_layout(showlegend=False)
    fig_dcw.update_layout(height=250, showlegend=False)
    fig_dcp.update_layout(height=250, showlegend=False)
    l_dch = df["volatility_dch"].iloc[-1]
    l_dcl = df["volatility_dcl"].iloc[-1]
    l_dcm = df["volatility_dcm"].iloc[-1]
    l_dcw = df["volatility_dcw"].iloc[-1]
    l_dcp = df["volatility_dcp"].iloc[-1]
    return df_dc, fig_dc, fig_dcw, fig_dcp, l_dch, l_dcl, l_dcm, l_dcw, l_dcp


df_dc, fig_dc, fig_dcw, fig_dcp, l_dch, l_dcl, l_dcm, l_dcw, l_dcp = donc(df)


@st.cache_resource
def kc(df):
    df_k = df.filter(['Close', "volatility_kcc",
                     "volatility_kch", "volatility_kcl"], axis=1)
#    df_k = df_k.dropna()
    df_k_ind = df.filter(["volatility_kchi", "volatility_kcli"], axis=1)
#    df_k_ind = df_k_ind.dropna()
    fig_k = px.line(df_k)
    fig_k.update_yaxes(title='Keltner Channels',
                       showticklabels=True, visible=True)
    fig_k2 = px.area(df["volatility_kcw"])
    fig_k3 = px.area(df["volatility_kcp"])
    fig_k2.update_yaxes(title='Keltner Bandwidth',
                        showticklabels=True, visible=True)
    fig_k3.update_yaxes(title='Keltner Percentile',
                        showticklabels=True, visible=True)
    l_kcp = df["volatility_kcp"].iloc[-1]
    l_kcw = df["volatility_kcw"].iloc[-1]
    l_kchi = df["volatility_kchi"].iloc[-1]
    l_kcli = df['volatility_kcli'].iloc[-1]
    fig_k.update_layout(showlegend=False)
    fig_k2.update_layout(showlegend=False)
    fig_k3.update_layout(showlegend=False)
    fig_k_ind = px.bar(df_k_ind)
    fig_k_ind.update_yaxes(title='Keltner Bandwidth & Hi-Lo',
                           showticklabels=True, visible=True)
    fig_k_ind.update_layout(showlegend=False)
    return df_k, df_k_ind, fig_k, fig_k2, fig_k_ind, fig_k3, l_kcp, l_kcw, l_kchi, l_kcli


df_k, df_k_ind, fig_k, fig_k2, fig_k_ind, fig_k3, l_kcp, l_kcw, l_kchi, l_kcli = kc(
    df)

# ############################################################
st.write("  --------------  ")
with st.empty():
    st.header("2. Analytics & Indicators")
with st.container():
    st.subheader("Trend Indicators")
    ct1, ct2, ct3, ct4, ct5, ct6 = st.columns(6)
    with ct1:
        st.metric("SMA", lsma_fast.round(2))
        st.metric("EMA", lema_fast.round(2))
    with ct2:
        st.metric("ADX", adx_last.round(2))
        st.metric("Mass Index", l_mi.round(2))
    with ct3:
        st.metric("Commodity Channel", l_cci.round(2))
        st.metric("TRIX", l_trix.round(2))
    with ct4:
        st.metric("Know Sure Index", l_kst.round(2))
        st.metric("Parbolic SAR", l_psar.round(2))
    with ct5:
        st.metric("Ichimoku Conv.", l_ichi.round(2))
        st.metric("Ichimoku Base", l_ichi_b.round(2))
    with ct6:
        st.metric("Schaff Trend Cycle", l_stc.round(2))
        st.metric("Detrended Pr Osc.", l_dp.round(2))
    st.info("Financial markets generally maintain consistent directional movement, in one direction or the other, over the longer term even as price exhibits randomness in shorter time frames. The indicators in this module exhibit the nature of price motion.")
with st.container():
    choix1 = st.selectbox("Select any option here!!",
                          trendlist, label_visibility="hidden")
    if choix1 == "Moving Averages (MA)":
        df_sma, lsma_fast, lsma_slow = sma(df)
        df_ema, lema_fast, lema_slow = ema(df)
        fig_ma = make_subplots(rows=1, cols=1, shared_xaxes=True)
        fig_ma.add_trace(go.Candlestick(x=df.index,
                                        open=df["Open"], high=df["High"],
                                        low=df["Low"], close=df["Close"],
                                        name="Price against Moving Averages",
                                        increasing_line_color='cyan',
                                        decreasing_line_color='gray'),
                         row=1, col=1)
        fig_ma.add_trace(go.Scatter(
            x=df.index, y=df['trend_sma_slow'], name='SMA 14 Days',
            showlegend=False), row=1, col=1)
        fig_ma.add_trace(go.Scatter(
            x=df.index, y=df['trend_ema_slow'], name='EMA 14 Days',
            showlegend=False), row=1, col=1)
        fig_ma.update_layout(xaxis_rangeslider_visible=False, showlegend=False)
        fig_sma = px.line(df_sma, height=300)
        fig_sma.update_layout(
            title='A - Simple Moving Average', showlegend=False)
        fig_sma.update_xaxes(visible=True, showticklabels=True)
        fig_sma.update_yaxes(title='SMA', visible=True, showticklabels=True)
        fig_ema = px.line(df_ema,  height=300)

        fig_ema.update_layout(
            title='B - Exponential Moving Average', showlegend=False)
        fig_ema.update_xaxes(visible=True, showticklabels=True)
        fig_ema.update_yaxes(title="EMA", visible=True, showticklabels=True)
        st.subheader('Moving Averages')
        ma3, ma4, ma5, ma6 = st.columns(4)
        with ma3:
            st.metric("Moving Average 21D", lsma_slow.round(2))
            del_sma = lsma_fast - lsma_slow
            st.metric("Spread", del_sma.round(2))
        with ma4:
            st.metric("Moving Average 14D", lsma_fast.round(2))
            if del_sma > 0:
                st.metric("SMA Signal", "Sell!!")
            elif del_sma == 0:
                st.metric("SMA Signal", "Reversal Ongoing!")
            else:
                st.metric("SMA Signal", "In Buy Zone")
        with ma5:
            st.metric("Exp.Moving Average 21D", lema_slow.round(2))
            del_ema = lema_fast - lema_slow
            st.metric("Spread", del_ema.round(2))
        with ma6:
            st.metric("Exp. Moving Average 14D", lema_fast.round(2))
            if del_ema > 0:
                st.metric("EMA Signal", "In Sell Zone")
            elif del_ema == 0:
                st.metric("EMA Signal", "Reaching Price Inflection!")
            else:
                st.metric("EMA Signal", "In Buy Zone")
        st.info("""Moving averages filter market noise and signal reversals through crossovers, over a period, thereby smoothing out the curve. Exponential Moving averages do the same with higher precision. When two MA/SMA or 2 SMA/EMA lines for the same period intersect, a reversal generally follows.""")
        st.plotly_chart(fig_ma, use_container_width=True)
        ma1, ma2 = st.columns(2)
        with ma1:
            st.info("""14 day and a 21 day Simple Moving Averages are plot over the Closing Prices of the security""")
            st.plotly_chart(fig_sma, use_container_width=True)
        with ma2:
            st.info("""14 day and a 21 day Exponential Moving Averages are plot over the Closing Prices of the security""")
            st.plotly_chart(fig_ema, use_container_width=True)
        
    elif choix1 == "Moving Average Convergence Divergence (MACD)":
        st.subheader("Moving Average Convergence-Divergence")
        fig_macd = px.area(df["trend_macd"])
        fig_macd.update_layout(title='MACD', showlegend=False)
        fig_macd.update_xaxes(visible=True, showticklabels=True)
        fig_macd.update_yaxes(title='MACD Signal',
                              visible=True, showticklabels=True)
        fig_macd_signal = px.bar(
            df["trend_macd_signal"], color=df["trend_macd_signal"])
        fig_macd_signal.update_layout(
            title='MACD Signal', height=300, showlegend=False)
        fig_macd_signal.update_xaxes(visible=True, showticklabels=True)
        fig_macd_signal.update_yaxes(
            title='MACD Signal', visible=True, showticklabels=True)

        fig_macd_diff = px.area(df["trend_macd_diff"])
        fig_macd_diff.update_layout(
            title='MACD Diff', height=300, showlegend=False)
        fig_macd_diff.update_xaxes(visible=True, showticklabels=True)
        fig_macd_diff.update_yaxes(
            title='MACD Diff', visible=True, showticklabels=True)
        mcd1, mcd2, mcd3 = st.columns([2, 1, 1])
        st.plotly_chart(fig_macd, use_container_width=True)
        st.info("The MACD Plot indicates that...")
        st.plotly_chart(fig_macd_diff, use_container_width=True)
        st.info("The MACD Diff Plot indicates that...")
        st.plotly_chart(fig_macd_signal, use_container_width=True)
        st.info("The MACD Signal Plot indicates that...")

    elif choix1 == "Average Directional Movement Index (ADX)":

        adx_last, fig_adx, fig_adx_sig = adx(df)
        adx1, adx2, adx3 = st.columns([2, 1, 1])
        with adx1:
            st.subheader("Average Directional Movement Index")
        with adx2:
            st.metric("ADX", adx_last.round(2))
        with adx3:
            st.write("*ADX SIGNAL*")
            if adx_last > 20:
                st.metric("Trend", "Strong")
            elif adx_last < 20:
                st.metric("Trend", "Weak")
            else:
                st.warning("Probable Sideways Trend")
        st.info("""The Average Directional Index is used to measure the strength of a trend, regardless of its direction. It is plotted as a single line with a value between 0 and 100.""")
        st.plotly_chart(fig_adx_sig, use_container_width=True)
        st.info("The Average Directional Index illustrates whether the Price Trend is going up or down in the current period, thereby enabling traders and investors to infer the upcoming scenarios and take their trading or investing decisions accordingly")

    elif choix1 == "Trix (TRIX)":

        l_trix, fig_trix, slope_trix = trix(df)
        tri1, tri2, tri3, tri4 = st.columns([1, 1, 1, 1])
        with tri1:
            st.subheader("TRIX")

        with tri2:
            st.metric("TRIX", l_trix.round(2))
        with tri3:
            if l_trix > 0:
                st.success("In Buy Zone!!")
                if slope_trix > 0:
                    st.info("Probable Buy")
                else:
                    st.warning("Probable Sell")

            elif l_trix < 0:
                st.error("In Sell Zone!!")
                if slope_trix > 0:
                    st.info("Probable Buy")
                else:
                    st.warning("Probable Sell")
            else:
                st.warning("Inflection ongoing !!")
        with tri4:
            if l_trix > 0:
                st.success("Momentum is Increasing!!")
            elif l_trix < 0:
                st.error("Momentum is Decreasing !!")
            else:
                st.warning("Inflection ongoing !!")
        st.info("The TRIX indicator, or Triple Exponential Moving Average, is a tool that measures the percentage change in a triple-smoothed exponential moving average.")
        st.plotly_chart(fig_trix, use_container_width=True)
        st.info("TRIX is an oscillator, meaning it oscillates around a zero line, and can be used as a momentum indicator. TRIX focuses on the rate of change of a triple exponential moving average, providing a unique perspective on price movements. It aims to filter out insignificant price fluctuations and highlight long-term trends")
    elif choix1 == "Mass Index (MI)":
        l_mi, fig_mi = mi(df)
        m2, m3 = st.columns([3, 1])
        with m2:
            st.subheader("Mass Index")
        with m3:
            st.metric("Mass Index", l_mi.round(2))
        st.info("The Mass Index  used to identify potential trend reversals in financial markets. It does this by measuring the expansion and contraction of price ranges. Essentially, it compares the average price range over time, looking for patterns that suggest a trend reversal might be imminent.")
        st.plotly_chart(fig_mi, use_container_width=True)
        st.info("""A 'reversal bulge' is a key signal. This occurs when the Mass Index rises above a certain value (like 27) and then falls back below a threshold (like 26.5). This suggests that the price range has expanded and then contracted, potentially indicating a trend reversal. The values here are unique for every security.""")

    elif choix1 == "Commodity Channel Index (CCI)":
        l_cci, fig_cci = cci(df)
        cc1, cc2, cc3 = st.columns([2, 2, 3])
        with cc1:
            st.subheader("Commodity Channel Index")
        with cc2:
            st.metric("Commodity Channel Index (CCI)", l_cci.round(2))
        with cc3:
            if l_cci > 0:
                st.metric("CCI", "Trading Above Mean")
            elif l_cci < 0:
                st.metric("CCI", "Trading Below Mean")
            else:
                st.warning("Prices are Average!!")
        st.info("""The Commodity Channel Index or CCI measure the current price's deviation from its average price over a specific period. It helps identify potential overbought or oversold conditions and can be used to spot price reversals or trend strength""")
        st.plotly_chart(fig_cci, use_container_width=True)
        st.info("""Above +100: Indicates that the current price is significantly above its average, suggesting a potentially overbought condition or a strong uptrend. Consequently, CCI below -100: Indicates that the current price is significantly below its average, suggesting an oversold condition or a strong downtrend. Outliers above and below +/- 100 indicate a strong reversal in the short term.""")

    elif choix1 == "Detrended Price Oscillator (DPO)":
        l_dp, fig_dpo = dpo(df)
        dp1, dp2 = st.columns([3, 1])
        with dp1:
            st.subheader("Detrended Price Oscillator")
        with dp2:
            st.metric("Detrended Price Oscillator (DPO)", l_dp.round(2))
        st.info('''Filter out the Long term trends and Gauge the spread between Price level Peaks and Troughs. The time between peaks and troughs in the DPO graph indicates the duration of these cycles''')
        st.plotly_chart(fig_dpo, use_container_width=True)
        st.info("""The Detrended Price Oscillator is designed to focus on short-term price cycles by filtering out long-term trends. It does this by shifting past price data against a Simple Moving Average (SMA). This approach helps identify periods of potential overbought or oversold conditions and the duration of price cycles.""")
    elif choix1 == "KST Oscillator (KST)":
        l_kst, fig_kst, fig_kst_sig, l_kst_sig = kst(df)
        ks1, ks3, ks4 = st.columns([2, 1, 1])
        with ks1:
            st.subheader("Know Sure Thing Index")
        with ks3:
            st.metric("KST Oscillator", l_kst.round(2))
        with ks4:
            if l_kst > l_kst_sig:
                st.metric("KST Signal", "In Buy Zone!!")
            else:
                st.error("In Sell Zone")
        st.info("The Know Sure Thing or KST Index is used to assess market trends and potential reversals. Developed by Martin Pring, it combines multiple smoothed rate-of-change calculations to provide a more comprehensive view of price momentum compared to individual rate-of-change calculations.")
        st.plotly_chart(fig_kst_sig, use_container_width=True)
        st.info("""A moving average of the KST line is often used as a signal line, which can help traders identify buy and sell signals. A positive and rising KST line suggests an uptrend, while a negative and falling KST line indicates a downtrend.""")
        st.info("""When the KST line crosses above its signal line from below, it can be interpreted as a buy signal. Conversely, a crossover of the KST line below its signal line from above can signal a sell opportunity""")
    elif choix1 == "Ichimoku Kinkō Hyō (Ichimoku)":
        fig_ichi, l_ichi, fig_ichi2, fig_ichi3 = ichi(df)
        ich1, ich2, ich3 = st.columns([3, 1, 1])
        with ich1:
            st.subheader("Ichimoku Kinkō Hyō")
        with ich2:
            st.metric("Ichimoku Index", l_ichi.round(2))
        with ich3:
            st.write("Signal in the Works!!")
        st.plotly_chart(fig_ichi, use_container_width=True)
        st.info("The Ichimoku Signal Indicates....")
        ic33, ic34 = st.columns(2)
        with ic33:
            st.plotly_chart(fig_ichi2, use_container_width=True)
        with ic34:
            st.plotly_chart(fig_ichi3, use_container_width=True)
        st.info("The Ichimoku Signal Indicates....")

    elif choix1 == "Parabolic Stop And Reverse (Parabolic SAR)":
        l_psar, fig_psar, l_psar_downi, l_psar_upi = psar(df)
        c44, c45, c46 = st.columns(3)
        with c44:
            st.subheader("Parabolic Stop And Reverse")
        with c45:
            st.metric("Parabolic SAR", l_psar.round(2))
        with c46:
            if l_psar_upi == 1:
                st.metric("PSAR Signal", f"{stock} in an Uptrend !!")
            elif l_psar_downi == 1:
                st.metric("PSAR Signal", f"{stock} in a Downtrend !!")
            else:
                st.metric("PSAR Signal", f"{stock} Trending Sideways")
        st.info("""The Parabolic SAR or PSAR Signal visually displays as a series of dots on a chart, which are positioned above or below the price. When the dots are below the price, it suggests an uptrend, while dots above the price indicate a downtrend. When the dots change position from below to above the price (or vice versa), it may signal a potential trend reversal.""")
        st.plotly_chart(fig_psar, use_container_width=True)
        st.info("""The Parabolic SAR or PSAR Signal can also be used as a dynamic stop-loss mechanism to protect unrealized gains""")

    else:
        l_stc, fig_stc = stc(df)
        st2, st3, st4 = st.columns(3)
        with st2:
            st.subheader("Schaff Trend Cycle (STC)")
        with st3:
            st.metric("STCI", l_stc.round(2))
        with st4:
            if l_stc > 75:
                st.metric("STCI Signal", "Inflection Approaching !!")
            elif l_stc < 20:
                st.metric("STCI Signal", "In Buy Zone !!")
            else:
                st.metric("STCI Signal", "Probable Sideways Movement")
        st.info("The Schaff Trend Cycle (STC) is used in trading and investing to identify trends and generate trading signals. It was developed to improve upon trading moving averages by incorporating cycle analysis")
        st.plotly_chart(fig_stc, use_container_width=True)
        st.info("""The STC is designed to identify trends and trend reversals by measuring the strength of the trend and the speed of price changes. The STC is an oscillator, which means that it measures the velocity of price movements.""")

st.write("  --------------  ")

with st.container():
    st.subheader("Momentum Indicators")
    ch1, ch2, ch3, ch4 = st.columns(4)
    with ch1:
        st.metric("Relative Strength Index", l_rsi.round(2))
        st.metric("Stochastic RSI", l_srsi.round(2))
    with ch2:
        st.metric("Ultimate Oscillator Index", l_uo.round(2))
        st.metric("Stochastic Oscillator Index", l_stoch.round(2))
    with ch3:
        st.metric("WIlliams %R", l_wr.round(2))
        st.metric("Kaufmann's MAverage", l_kama.round(2))
    with ch4:
        st.metric("True Strength Index", l_tsi.round(2))
        st.metric("Rate of Change", l_roc.round(2))
st.info(""" Assess the speed, momentum and direction of price changes for an
asset, gauge the asset's inertia.""")

with st.container():
    choix2 = st.selectbox("Select any option here!!",
                          momenlist, label_visibility="hidden")
    if choix2 == "Relative Strength Index (RSI)":
        fig_rsi, l_rsi, l_rsi21, fig_rsi_b = rsif(df)
        del_rsi = l_rsi - 50
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.subheader("Relative Strength Index")
        with r2:
            st.metric("Relative Strength", l_rsi.round(2),
                      delta=del_rsi.round(2), delta_color="inverse")
        with r3:
            if l_rsi21 > 70:
                st.metric("RSI Indicator", "Overbought")
            elif l_rsi21 < 30:
                st.metric("RSI Indicator", "Oversold")
            else:
                st.metric("RSI Indicator", "In Trading")
        with r4:
            if l_rsi21 > l_rsi:
                st.metric("RSI Signal", "Release")
            elif l_rsi21 < l_rsi:
                st.metric("RSI Signal", "Purchase")
            else:
                st.warning("Inflection ongoing!!")
        st.info("""The RSI Indicator enables one to measure the rate of change
in a stock's price over time, accounting in volume or in a momentum metric,
in other words""")
        st.plotly_chart(fig_rsi, use_container_width=True)
        st.info("""Investigate whether an asset has been Overbought or Oversold,
and to what level. An RSI of above 70 indicates an Overbought Condition, and one below 30 indicates that it is Oversold. The Zero line seperates the Buy and Sell regions.""")
        st.plotly_chart(fig_rsi_b, use_container_width=True)
    elif choix2 == "Stochastic RSI (SRSI)":
        l_srsi, fig_rsik, l_srsi_k, l_srsi_d, fig_rsik2 = srsi(df)
        c46, c47, c48, c49 = st.columns(4)
        with c46:
            st.subheader("Stochastic RSI")
        with c47:
            st.metric("Stoch RSI", round(l_srsi))
        with c48:
            if l_srsi > 0.75:
                st.metric("Price Level", "Overbought")
            elif l_srsi < 0.25:
                st.metric("Price Level", "Oversold")
            else:
                st.metric("Price Level", "Moderate")
        with c49:
            if l_srsi_k > l_srsi_d:
                st.metric("Trading Indicator", "Purchase!")
            else:
                st.metric("Trading Indicator", "Release!")
        st.info("""The Stochastic RSI (StochRSI) is a technical analysis
indicator that measures the rate of change of the Relative Strength Index (RSI),
providing an oscillator that fluctuates between 0 and 100.""")
        st.plotly_chart(fig_rsik, use_container_width=True)
        st.plotly_chart(fig_rsik2, use_container_width=True)
        st.info("""The SRSI Indicator is used in parallel to other indicators,
like the RSI, TSI or MACD""")

    elif choix2 == "True Strength Index (TSI)":
        fig_tsi, l_tsi, hi_lim_tsi, lo_lim_tsi = tsi(df)
        ts1, ts2, ts3 = st.columns(3)
        with ts1:
            st.subheader("True strength index")
        with ts2:
            st.metric("True Strength Index", l_tsi.round(2))
        with ts3:
            if l_tsi > hi_lim_tsi:
                st.metric("TSI Signal", "Overbought!!")
            elif l_tsi < lo_lim_tsi:
                st.metric("TSI Signal", "Oversold!!")
            else:
                st.metric("TSI Signal", "Actively Trading!")
        st.info("""The True Strength Indicator [TSI] Indicator TSI is
calculated by smoothing the price change and then smoothing the absolute price
change. The TSI value is then derived by dividing the smoothed price change by
the smoothed absolute price change, thereby cutting out the noise""")
        st.plotly_chart(fig_tsi, use_container_width=True)

        st.info("""The TSI is a valuable tool for technical analysis, but it's
important to use it in conjunction with other indicators and to be aware of its
limitations""")

    elif choix2 == "Ultimate Oscillator (UO)":
        l_uo, fig_uo = uo(df)
        uoc1, uoc2, uoc3 = st.columns(3)
        with uoc1:
            st.subheader("Ultimate Oscillator")
        with uoc2:
            st.metric("Ultimate Oscillator Index", l_uo.round(2))
        with uoc3:
            if l_uo>70:
                st.metric("Status", "Overbought")
            elif l_uo<30:
                st.metric("Status", "Oversold")
            else:
                st.metric("Status", "Actively Trading")
        st.info("""The UO calculates the buying pressure (the difference between
the closing price and the lowest low) and the true range (the highest price
minus the lowest price) over three different time periods (e.g., 7, 14, and 28
periods). It then uses these values to calculate a weighted average, with the
shorter periods given more weight, ultimately resulting in a UO value between 0
and 100. """)
        st.plotly_chart(fig_uo, use_container_width=True)
        st.info("""The Ultimate Oscillator/Indicator, as mentioned above,
measures buying pressure across three different timeframes (short, medium, and
long) to identify potential overbought and oversold conditions. Unlike
traditional oscillators that rely on a single timeframe, the UO combines these
timeframes, aiming to provide a more reliable momentum reading""")

    elif choix2 == "Stochastic Oscillator (SR)":
        st.subheader("Stochastic Oscillator")
        st.info("""The Stochastic Oscillator measures the relationship between
a security's current closing price and its price range over a specific period.
It's a momentum indicator used to identify potential overbought or oversold
conditions and spot trend reversals.""")
        fig_stoch, fig_stoch_sig, l_stoch = stoch(df)
        st.info("""Values above 80 are often considered overbought, suggesting a
potential sell signal, while values below 20 are considered oversold, suggesting
a potential buy signal. """)
        st.plotly_chart(fig_stoch, use_container_width=True)
        st.plotly_chart(fig_stoch_sig, use_container_width=True)
    elif choix2 == "Williams %R (WR)":
        l_wr, fig_wr = wr(df)
        wrc1, wrc2, wrc3 = st.columns(3)
        with wrc1:
            st.subheader("Williams %R")
        with wrc2:
            st.metric('Williams R Percentage', l_wr.round(2))
        with wrc3:
            if l_wr < -80:
                st.metric("WR Signal", "Oversold")
            elif l_wr > -20:
                st.error("WR Signal", "Overbought")
            else:
                st.warning("""Refer to Indicator value to observe if its closer
to 0 or otherwise""")
        st.info("""The Williams %R Indicator maps momentum for a particular
security, and it helps estimate the condition of the security in terms of
demand""")
        st.plotly_chart(fig_wr, use_container_width=True)
        st.info("""The Williams R Percentage Indicator ranges from 0 to -100,
with values closer to 0 suggesting overbought conditions and values closer to
-100 suggesting oversold conditions.""")

    elif choix2 == "Awesome Oscillator (AO)":
        l_ao, fig_ao = ao(df)
        ao1, ao2, ao3 = st.columns(3)
        with ao1:
            st.subheader("Awesome Oscillator")
        with ao2:
            st.metric("AO", l_ao.round(2))
        with ao3:
            if l_ao > 0:
                st.success("In Buy Zone!!")
            elif l_ao < 0:
                st.warning("In Sell Zone!!")
            else:
                st.warning("Inflection Ahead!!")


        st.info("""The Awesome Oscillator compares a 5-period Simple Moving
Average (SMA) to a 34-period SMA, both applied to the midpoints of candlesticks,
to measure the difference between short-term and long-term price movements. """)
        st.plotly_chart(fig_ao, use_container_width=True)
        st.info("""When the AO is above the zero line, it suggests a bullish
trend, indicating that short-term momentum is rising faster than long-term
momentum.Zero Line Cross: When the AO crosses above the zero line, it's
considered a potential bullish signal, and when it crosses below, it's a
potential bearish signal.
 """)
    elif choix2 == "Kaufmans Adaptive Moving Average (KAMA)":
        st.subheader("Kaufmans Adaptive Moving Average")
        fig_kama, l_kama = kama(df)
        st.info("""KAMA diverges away for a change in a securitys price. When
market volatility is low, Kaufman’s Adaptive Moving Average remains near the
current market price, but when volatility increases, it will lag behind.""")
        st.write("Map trend, time turning points and filter price movements.")
        st.plotly_chart(fig_kama, use_container_width=True)
        st.info("""Unlike other moving averages, Kaufman’s Adaptive Moving
Average accounts not only for price action but also for market volatility.""")
        st.info("""Basically, when the KAMA indicator line is moving lower, it
indicates the existence of a downtrend. On the other hand, when the KAMA line is
moving higher, it shows an uptrend. As compared to the Simple Moving Average,
the KAMA indicator is less likely to generate false signals that may cause a
trader to incur losses.""")
    elif choix2 == "Rate of Change (ROC)":
        l_roc, fig_roc = roc(df)
        rc11, rc12 = st.columns([2, 1])
        with rc11:
            st.subheader("Rate of Change")
        with rc12:
            st.metric("Rate of Change", l_roc.round(2))
        st.plotly_chart(fig_roc, use_container_width=True)
        st.info(""" Rate of Change oscillator, is a momentum indicator in
technical analysis that measures the percentage change in price over a defined
period. It's often used to identify potential overbought and oversold
conditions, spot divergences, and confirm trends. A Positive ROC means an
upward trend, and a negative ROC, therefore, means a downward trend in the
security's price""")
    elif choix2 == "Percentage Price Oscillator (PPO)":
        st.subheader("Percentage Price Oscillator")
        fig_ppo, fig_ppo_signal = ppo(df)
        st.info("""measures the percentage difference between two Exponential
Moving Averages (EMAs), typically a 9-day EMA and a 26-day EMA. It's calculated
by subtracting the slower EMA from the faster EMA and then dividing the result
by the slower EMA. """)
        st.plotly_chart(fig_ppo, use_container_width=True)
        st.info("""A positive PPO value indicates a bullish trend (short-term
EMA is rising faster than the long-term EMA), conversely a negative PPO
indicates a bearish trend (short-term EMA is declining faster than the long-term
EMA). Zero Line crossovers indicate trend reversals. """)
        # st.plotly_chart(fig_ppo_hist, use_container_width=True)
    else:
        st.subheader("Percentage Volume Oscillator (PVO)")
        fig_pvo, fig_pvo_signal = pvo(df)
        st.markdown("""PVO measures the momentum of volume changes by comparing
two volume-based moving averages. It calculates the percentage difference
between a short-term and long-term moving average of volume, providing insights
into the strength and direction of volume trends""")
        st.plotly_chart(fig_pvo, use_container_width=True)
        st.info("""The PVO fluctuates around the zero line. When the PVO is
above zero, the short-term volume is higher than the long-term volume, and vice
vers.""")
        st.info("""Crosses over or below the Zero Line indicates the momentum
of the security's price trajectory reversal. The PVO is often plotted with a
signal line, which is a smoothed version of the PVO itself. This can help
identify potential entry and exit points. """)

st.write("  --------------  ")

with st.container():
    cvx1, cvx2, cvx3, cvx4, cvx5 = st.columns([2, 1, 1, 1, 1])
    with cvx1:
        st.header("Volatility Indicators")
    with cvx2:
        st.metric("Avg True Range", l_atr.round(2))
        st.metric("Ulcer Index", l_ui.round(2))
    with cvx3:
        st.metric("Bollinger %tile", l_bbp.round(2))
        st.metric("Bollinger Squeeze", l_bbw.round(2))
    with cvx4:
        st.metric("Keltner %tile", l_kcp.round(2))
        st.metric("Keltner Squeeze", l_kcw.round(2))
    with cvx5:
        st.metric("Donchian %tile", l_dcp.round(2))
        st.metric("Donchian Squeeze", l_dcw.round(2))

st.info(""" Volatility indicators are tools traders use to
measure the intensity or magnitude of price fluctuations in financial
instruments.""")

with st.container():
    choix3 = st.selectbox("Select any option here!!",
                          volatlist, label_visibility="hidden")
    if choix3 == "Average True Range (ATR) & Ulcer Index (UI)":
        att12, at12, att13, att14, att15 = st.columns([2, 1, 1, 1, 3])
        with att12:
            st.subheader("Average True Range")
            st.subheader("Ulcer Index")
        with att13:
            st.metric("ATR [INR]", l_atr.round(2))
        with at12:
            st.metric("Ulcer Index", l_ui.round(2))
        with att14:
            if l_atr > 40:
                st.metric("Volatility - ATR", "Medium")
            if l_atr < 21:
                st.metric("Volatility - ATR", "Low")
            else:
                st.metric("Volatility - ATR", "High")
            # else: st.write("Volatility", "Unchartable")
        with att15:
            if l_ui < 2.5:
                st.metric("Ulcer Signal", "Nearing Prev High!!")
            elif l_ui > 5:
                st.metric("Ulcer Signal", "Far from Prev High")
            else:
                st.metric("Ulcer Signal", "Median Levels")
        st.info(""" Average True Range or ATR is used to measure market
volatility. It's calculated by averaging the true range, which considers the
price movement between the day's high and low, and the previous day's closing
price, over a specified period, typically 14 days.""")
        st.plotly_chart(fig_atr, use_container_width=True)
        st.info("""The Ulcer Index is a popular technical indicator that lets
one know the maximum downside risk before entering a trade.""")
        st.plotly_chart(fig_ui, use_container_width=True)
    elif choix3 == "Bollinger Bands (BB)":
        df_bb1, df_bb2, df_bb_ind, l_bbp, fig_bb, fig_bb2, l_bbhi, l_bbli, l_bbw = bb(
            df)
        bx2, bx3, bx4, bx5 = st.columns(4)
        with bx2:
            st.subheader("Bollinger Bands")
        with bx3:
            if l_bbhi == 1:
                st.metric("BB Hi Signal", "Nearing Highs>Sell!")
            else:
                st.metric("BB Hi Signal", "Stable")
        with bx4:
            if l_bbli == 1:
                st.metric("BB Lo Signal", "Nearing Lows>Acquire!")
            else:
                st.metric('BB Lo Signal', "None")
        with bx5:
            if l_bbp > 0.7:
                st.metric("Percentile Signal", "Around the Highs")
            elif l_bbp < 0.3:
                st.metric("Percentile Signal", "Around the Lows")
            else:
                st.metric("Percentile Signal", "Around the Mids")
        st.info(""" Bollinger's Bands plot measures market volatility
and potential price reversals. They consist of a moving average (middle band),
an upper band (moving average + a multiple of standard deviation), and a lower
band (moving average - a multiple of standard deviation), typically 2 standard
deviation""")
        st.plotly_chart(fig_bb, use_container_width=True)
        st.info(""" Bollinger's Bands Percentile Indicator plots measures the
relative width of the Bollinger Bands compared to their historical range. It
helps traders gauge the current volatility level and identify potential trend
reversals or continuation signals,""")
        st.plotly_chart(fig_bb2, use_container_width=True)
        st.info("""The Bollinger Band Squeeze identifies periods of low
volatility (band contraction) that often precede significant price movements,
making it an effective tool for anticipating breakouts""")
    elif choix3 == "Keltner Channel (KC)":
        df_k, df_k_ind, fig_k, fig_k2, fig_k_ind, fig_k3, l_kcp, l_kcw, l_kchi, l_kcli = kc(
            df)
        kcx1, kcx2, kcx3, kcx4, kcx5 = st.columns(5)
        with kcx1:
            st.subheader("Keltner Channels")
        with kcx2:
            st.metric('Keltner Squeeze', l_kcw.round(2))
        with kcx3:
            st.metric("Keltner %tile", l_kcp.round(2))
        with kcx4:
            if l_kchi == 1:
                st.metric("Keltner Hi Signal", "In Buy Zone!!")
            else:
                st.metric("Keltner Hi Signal", "None")
        with kcx5:
            if l_kcli == 1:
                st.metric("Keltner Lo Signal", "In Sell Zone")
            else:
                st.metric("Keltner Lo Signal", "None")
        st.plotly_chart(fig_k, use_container_width=True)
        st.info("""Similar to Bollinger's Bands, Keltner Channels indicate
the ranges with a volatility-based metric that identifies an upper and lower
boundary for price movements, based on recent chart data.""")
        st.info ("""If theprice of an asset moves to the upper bound of the
channel, traders can sell the asset knowing that the price is likely to revert
to the mean.""")
        st.plotly_chart(fig_k2, use_container_width=True)
        st.info("""Keltner Bandwidth helps estimate the Volatility ranges of
any security""")
        st.plotly_chart(fig_k3, use_container_width=True)
        st.info("""Kelner Squeeze is a factor which merges with the Keltner
function to help one identify market consolidation and anticipate potential
breakouts""")
        st.plotly_chart(fig_k_ind, use_container_width=True)
        st.info("Keltner Bandwidth & Hi-Lo")
    else:
        df_dc, fig_dc, fig_dcw, fig_dcp, l_dch, l_dcl, l_dcm, l_dcw, l_dcp = donc(
            df)
        dcc1, dcc2, dcc3, dcc4, dx5 = st.columns(5)
        with dcc1:
            st.subheader("Donchian Channels")
        with dcc2:
            st.metric("Donchian High", l_dch.round(3))
        with dcc3:
            st.metric("Donchian Med", l_dcm.round(3))
        with dcc4:
            st.metric("Donchian Low", l_dcl.round(3))
        with dx5:
            if l_bbp > 0.7:
                st.metric("%-tile Signal", "*High Region*")
            elif l_bbp < 0.3:
                st.metric("%-tile Signal", "*Low Region*")
            else:
                st.metric("%-tile Signal", "*Mid Region*")
        st.info("""Donchian Channels are used to determine the relative
volatility of a market and the potential for price breakouts. The area between
the upper and lower bands is called the Donchian channel, wherein each band
is constructed by the Moving Averages of the Highs and Lows.""")
        st.plotly_chart(fig_dc, use_container_width=True)
        st.info("""Gauge the spread from the Donchian Median Levels .""")
        st.plotly_chart(fig_dcw, use_container_width=True)
        st.info("""Calculate Donchian Channels using the 100th and 0th
percentile ranks.""")
        st.plotly_chart(fig_dcp, use_container_width=True)


# ----------------------###---------------------#####
st.write("  --------------  ")
with st.container():
    l_dr, fig_dr, l_dlr, delta_r, fig_dlr = dr(df)
    l_cr, fig_cr = cr(df)
    cr1, cr2, cr3, cr4, cr5 = st.columns([1, 1, 1, 1, 1])
    with cr1:
        st.header("Returns Indicators")
    with cr2:
        st.metric("Cumulative Returns", l_cr.round(2))
    with cr3:
        # st.metric("Direct Returns", l_dr.round(2))
        st.metric("Log Direct Returns", l_dlr.round(2))
    with cr4:
        st.metric("Spread against DR", delta_r.round(2))
    with cr5:
        if delta_r > 0:
            sign_r = "Positive"
        elif delta_r < 0:
            sign_r == "Negative"
        else:
            sign_r == "Inflection!"
        st.metric("Signal", sign_r)
    st.info("""A returns percentage indicator" refers to a measure that
expresses the gain or loss of an investment or project over a period as a
percentage of the initial investment.""")
    st.plotly_chart(fig_dr, use_container_width=True)
    st.plotly_chart(fig_dlr, use_container_width=True)
    st.info("""The Returns Index is plotted over the Logarithmic Returns. If
the Returns are below the Log Returns, then the asset or security is
Profitable. Conversely, If it is above the logarithmic curve, it records
positive Returns on investment""")
st.write("  --------------  ")
with st.empty():
    cvv1, cvv2, cvv3, cvv4, cvv5 = st.columns(5)
    with cvv1:
        st.header("Volume Indicators")
    with cvv2:
        st.metric("Money Flow Index", l_mfi.round(2))
        st.metric("On Balance Volume [x100k]", l_obv.round(2))

    with cvv3:
        st.metric("ADI [x100k]", l_adi.round(2))
        st.metric("Chaikin's Money Flow Index", l_cmf.round(2))

    with cvv4:
        lfi = l_fi/10000
        lvpt = l_vp/10000
        st.metric("Force Index [x10k]", lfi.round(2))
        st.metric("Volume Price Trend [x10k]", lvpt.round(2))

    with cvv5:
        st.metric("Ease of Movement", l_em.round(2))
        st.metric("Negative Volume Index", l_nvi.round(2))

with st.container():
    choix5 = st.selectbox("Select any option here!!",
                          volulist, label_visibility="hidden")
    if choix5 == "Money Flow Index (MFI)":
        l_mfi, fig_mfi = mfi(df)
        mfi1, mfi2, mfi3 = st.columns([3, 1, 1])
        with mfi1:
            st.subheader("Money Flow Index")
        with mfi2:
            st.metric("Money Flow", l_mfi.round(2))
        with mfi3:
            if l_mfi < 10:
                st.metric("Trading Zone", "Long")
            else:
                st.metric("Trading Zone", "Short")
        st.info("""The MFI oscillates between 0 - 100. It helps traders
identify overbought and oversold conditions, as well as detect potential
trend reversals""")
        st.markdown("""MFI is  a volume-weighted version of the Relative
Strength Index (RSI).""")
        st.plotly_chart(fig_mfi, use_container_width=True)
        st.info(""" By incorporating volume, the MFI provides a more
           comprehensive picture of market sentiment than price-only
          indicators""")
    elif choix5 == "Accumulation/Distribution Index (ADI)":
        l_adi, fig_adi = adi(df)
        ad1, ad2 = st.columns([3, 1])
        with ad1:
            st.subheader("Accumulation/Distribution Index")
            st.markdown(
                """
                1. Money Flow Multiplier = [(Close  -  Low) - (High - Close)]/(High - Low)
                2. Money Flow Volume = Money Flow Multiplier x Volume for the Period
                3. ADL = Previous ADL + Current Period's Money Flow Volume
                """
                )
        with ad2:
            st.metric("Calculated", l_adi.round(2))
        st.info("The Accumulation/Distribution (A/D) indicator is a volume-based technical analysis tool that helps identify buying and selling pressure, and assess the strength of a trend in a security. It analyzes the relationship between price and volume to determine if a stock is being accumulated (buying pressure) or distributed (selling pressure)")
        st.plotly_chart(fig_adi, use_container_width=True)
        st.info("The Accumulation Distribution Line is a running total of each period's Money Flow Volume. First, a multiplier is calculated based on the relationship of the close to the high-low range. Second, the Money Flow Multiplier is multiplied by the period's volume to come up with a Money Flow Volume. A running total of the Money Flow Volume forms the Accumulation Distribution Line. ")
        ad3, ad4 = st.columns(2)
        with ad3:
           st.markdown("""A high positive multiplier combined with high volume
             shows strong buying pressure that pushes the indicator higher.
             Conversely, a low negative number combined with high volume
             reflects strong selling pressure that pushes the indicator
             lower.""")
        with ad4:
            st.markdown("""An uptrend in prices with a downtrend in the AD Line
            suggests underlying selling pressure (distribution) that could
            foreshadow a bearish reversal on the price chart. A downtrend in
            prices with an uptrend in the Accumulation Distribution Line
            indicates underlying buying pressure (accumulation) that could
foreshadow a bullish price reversal.""")
    elif choix5 == "On-Balance Volume (OBV)":
        l_obv, fig_obv = obvf(df)
        ob1, ob2 = st.columns([3, 1])
        with ob1:
            st.subheader("On-Balance Volume (OBV)")
            st.write("On Balance Volume provides a cumulative measure of an asset's volume, adding to it when the price goes up and subtracting from it when the price goes down")
        with ob2:
            st.metric("On balance volume [x100000]", l_obv.round(2))
        st.plotly_chart(fig_obv, use_container_width=True)
        st.info("When both price and OBV are making higher peaks and higher troughs, the upward trend is likely to continue, and consequently, When both price and OBV are making lower peaks and lower troughs, the downward trend is likely to continue.")
        st.info("During a trading range, if the OBV is rising, accumulation may be taking place—a warning of an upward breakout, and in case it is falling, there are chances of a downward breakout.")

    elif choix5 == "Chaikin Money Flow (CMF)":
        l_cmf, fig_cmf = cmf(df)
        cm1, cm2, cm3 = st.columns([2, 1, 1])
        with cm1:
            st.subheader("Chaikin Money Flow Index")
            st.write("Chaikin Money Flow values between 0.05 – 0.25 indicate buying pressures from the bulls and values between -0.05 to -0.25 indicate selling pressures from the bears.")
        with cm2:
            st.metric("CMFI", l_cmf.round(2))
        with cm3:
            if l_cmf > 0.04:
                st.metric("Market Pressure", "Purchase!!")
            elif l_cmf < -0.04:
                st.metric("Market Pressure", "Liquidate!!")
            else:
                st.warning("In Active Trading Domain!!")
        st.plotly_chart(fig_cmf, use_container_width=True)
        st.info("The Chaikin Money Flow (CMF) is a volume-weighted average of accumulation and distribution over a specified period. The standard CMF period is 21 days.")
        st.info("The principle behind the Chaikin Money Flow is the nearer the closing price is to the high, the more accumulation has taken place. Conversely, the nearer the closing price is to the low, the more distribution has taken place. If the price action consistently closes above the bar's midpoint on increasing volume, the Chaikin Money Flow will be positive. Conversely, if the price action consistently closes below the bar's midpoint on increasing volume, the Chaikin Money Flow will be a negative value.")
    elif choix5 == "Force Index (FI)":
        l_fi, fig_fi = fi(df)
        l_fi2 = l_fi/1000000
        fo1, fo2 = st.columns([3, 1])
        with fo1:
            st.subheader("Force Index")
        with fo2:
            st.metric("Force Index - Millions", l_fi2.round(2))
        st.info("The Force Index is calculated by subtracting yesterday's close from today's close and multiplying the result by today's volume. If closing prices are higher today than yesterday, the force is positive. If closing prices are lower than yesterday's, the force is negative.")
        st.plotly_chart(fig_fi)
    elif choix5 == "Ease of Movement (EoM, EMV)":
        fig_em, fig_sma_em, l_em, l_sma_em, delta_em = em(df)
        sm3, sm4, sm5, sm6 = st.columns(4)
        with sm3:
            st.subheader("Ease of Movement Index")
        with sm4:
            st.metric("Ease-of-Movement Index", l_em.round(2))
        with sm5:
            st.metric("EOM SMA", l_sma_em.round(2))
        with sm6:
            if delta_em == 0:
                st.metric("EM Signal", "Reversal Incoming!!")
            else:
                st.metric("EM Signal", "No Reversals yet!")
        st.info("The Ease of Movement EoM or EMV Indicator measures the relationship between price change and volume change. It helps traders assess the strength of price movements, particularly whether a market is trending or consolidating. ")
        st.plotly_chart(fig_em, use_container_width=True)
        st.info("A rising EOM suggests that prices are moving upward with relative ease, potentially indicating a strong upward trend, conversely, A falling EOM may indicate that prices are moving downward more easily, potentially signaling a weakening trend.A cross above zero suggests that prices are moving upward with more ease, while a cross below zero suggests downward movement with more ease ")
    elif choix5 == "Volume-price Trend (VPT)":
        l_vp, fig_vpt = vpt(df)
        vp1, vp2 = st.columns([3, 1])
        with vp1:
            st.subheader("Volume-price Trend")
            st.write(
                '''Determine the balance between an assets
                demand and supply.''')

        with vp2:
            st.metric("VPTI", l_vp.round(2))
        st.plotly_chart(fig_vpt, use_container_width=True)
        st.info("A rising VPT with increasing volume suggests a strong bullish trend, and a falling VPT with decreasing volume suggests a strong bearish trend")
        st.info("In congruence, A rising VPT with decreasing volume might indicate a weakening trend, whereas, a falling VPT with increasing volume might indicate a weakening trend in reverse, thus enabling one to estimate Divergences between the price and VPT can signal potential trend reversals")
    elif choix5 == "Negative Volume Index (NVI)":
        l_nvi, fig_nvi = nvi(df)
        nv1, nv2 = st.columns([3, 1])
        with nv1:
            st.subheader("Negative Volume Index (NVI)")

        with nv2:
            st.metric("NVI", l_nvi.round(2))
        st.write("The Negative Volume Index (NVI) is a technical indicator that helps traders identify potential trend reversals or continuations by analyzing volume changes, particularly when volume decreases.")
        st.plotly_chart(fig_nvi, use_container_width=True)
        st.info("A rising NVI may indicate that smart money is accumulating, potentially supporting a bullish trend. A declining NVI might suggest institutional selling or bearish sentiment. Furthermore, The NVI focuses on periods of reduced market activity, thereby filtering out the noise from high-volume day.")
    else:
        l_vw, fig_vwap = vwap(df)
        vv1, vv2 = st.columns([3, 1])
        with vv1:
            st.subheader("Volume Adjusted Average Price")
        with vv2:
            st.metric("Volume Volume-Weighted Price", l_vw.round(2))
        st.write("Used to assess the average price of a security over a specific period, considering both the price and the volume of trades, VWAP is calculated by multiplying the price of each trade by its volume, summing these products, and then dividing by the total volume of trades during that period")
        st.plotly_chart(fig_vwap, use_container_width=True)
        st.info("VWAP, or Volume-Weighted Average Price, is a technical indicator used in intraday charts to determine the average price of a security based on volume and price. It's essentially a benchmark for determining if a stock is undervalued or overvalued during a trading day.")
st.write("  --------------  ")
with st.container():
    st.markdown(''' <div align="center"><h2>3. Company Information,
                Data & Statistics</h1></div>''',
                unsafe_allow_html=True)
    sc1, sc2 = st.columns([2, 1])
    with sc1:
        st.subheader("Business Description")
        st.info(dfi.at['longBusinessSummary', 'Details'])
        try:
            floatShares = dfi.at['floatShares', 'Details']  # 7A_C
            sharesOutstanding = dfi.at['sharesOutstanding', 'Details']
            impliedSharesOutstanding = dfi.at['impliedSharesOutstanding',
                                              'Details']
            fs1, fs2 = st.columns(2)
            with fs1:
                st.write("Floating Shares", floatShares)
            with fs2:
                st.write("Outstanding Shares", sharesOutstanding)
            fig_shares = px.pie(values=[floatShares, sharesOutstanding,
                                        impliedSharesOutstanding],
                                names=['Floating', 'Outstanding', 'Implied'],
                                hole=0.45)
            fig_shares.update_layout(
                title='Shares & Stocks Issued', title_x=0.33,
                showlegend=False)
            st.plotly_chart(fig_shares, use_container_width=True)
        except Exception:
            st.write("Data Unreported")
        try:
            heldPercentInsiders = dfi.at['heldPercentInsiders', 'Details']
            heldPercentInstitutions = dfi.at['heldPercentInstitutions',
                                             'Details']
            fig_holdings = px.pie(
                values=[heldPercentInsiders, heldPercentInstitutions],
                hole=0.45)
            fig_holdings.update_layout(
                title='Stakeholding Pattern', title_x=0.33,
                showlegend=True)
            st.plotly_chart(fig_holdings, use_container_width=True)
            hx1, hx2 = st.columns(2)
            with hx1:
                st.write("Promoters", heldPercentInsiders)
            with hx2:
                st.write("Institutions", heldPercentInstitutions)
        except Exception:
            st.write("Data Unreported")
        try:
            overallRisk = dfi.at['overallRisk', 'Details']
            st.write("Overall Institutional Risk", overallRisk)
            auditRisk = dfi.at['auditRisk', 'Details']  # 6c
            #           st.write("Audit Risk", auditRisk)
            boardRisk = dfi.at['boardRisk', "Details"]
            #            st.write("Board Risk", boardRisk)
            compensationRisk = dfi.at['compensationRisk', 'Details']
            shareHolderRightsRisk = dfi.at['shareHolderRightsRisk', 'Details']
            df_risk = pd.DataFrame({'Risk Types': ["Audit Risk", "Board Risk",
                                                   "Stakeholders' Rights Risk",
                                                   "Compensation Risk"],
                                    'Risk Levels': [auditRisk, boardRisk,
                                                    shareHolderRightsRisk,
                                                    compensationRisk]})
            figure_risk = px.bar(
                df_risk["Risk Levels"], labels=df_risk['Risk Types'],
                color=df_risk['Risk Types'])
            figure_risk.update_layout(
                height=300, title='Institutional Risk Map',  title_x=0.25,
                showlegend=False)

            figure_risk.update_xaxes(showticklabels=True, visible=True)
            figure_risk.update_yaxes(showticklabels=True, visible=True)
            #          st.write("Compensation Risk", compensationRisk)
            #      st.write("Shareholders' Rights Risk", shareHolderRightsRisk)
            st.plotly_chart(figure_risk, use_container_width=True)
        except Exception:
            st.write("Data Unreported")
        st.write("  ----------  ")
    try:
        with st.expander("Find Details on Company Officers here..."):
            st.write(dfi.at['companyOfficers', 'Details'])
    except Exception:
        st.write("Data Unreported")
    with sc2:
        st.subheader("Volume Metrics")
        try:
            averageVolume = dfi.at['averageVolume', 'Details']  # 6a
            st.write("Mean Volume - ", averageVolume)
        except Exception:
            st.write("Data Unreported")
        try:
            regularMarketVolume = dfi.at['regularMarketVolume', 'Details']
            st.write("Regular Market Volume - ", regularMarketVolume)
        except Exception:
            st.write("Data Unreported")
        try:
            averageVolume10days = dfi.at['averageVolume10days', 'Details']
            st.write('10Day Volume Avg - ', averageVolume10days)
        except Exception:
            st.write("Data Unreported")
        try:
            averageDailyVolume10Day = dfi.at['averageDailyVolume10Day',
                                             'Details']
            st.write('10D Daily Volume - ', averageDailyVolume10Day)
        except Exception:
            st.write("Data Unreported")
        st.write("  --------  ")
        ba1, ba2 = st.columns(2)
        with ba1:
            try:
                bid = dfi.at['bid', 'Details']  # 6b
                st.write("Bids - ", bid)
                bidsize = dfi.at['bidSize', 'Details']
                st.write("Bid Sizes - ", bidsize)
            except Exception:
                st.write("Data Unreported")
        with ba2:
            try:
                ask = dfi.at['ask', 'Details']
                st.write("Asks - ", ask)
                asksize = dfi.at['askSize', 'Details']
                st.write("Ask Sizes - ", asksize)
            except Exception:
                st.write("Data Unreported")
        st.write("  ----------  ")
        st.subheader("Financial Insights")
        try:
            totalCash = dfi.at['totalCash', 'Details']
            st.write("Total Cash", totalCash)
        except Exception:
            st.write("Data Unreported")
        try:
            totalCashPerShare = dfi.at['totalCashPerShare', 'Details']
            st.write("Total Cash/Share", totalCashPerShare)
        except Exception:
            st.write("Data Unreported")
        try:
            totalDebt = dfi.at['totalDebt', 'Details']
            st.write("Total Debt", totalDebt)
        except Exception:
            st.write("Data Unreported")
        try:
            totalRevenue = dfi.at['totalRevenue', 'Details']
            st.write("Total Revenue", totalRevenue)
        except Exception:
            st.write("Data Unreported")
        try:
            revenuePerShare = dfi.at['revenuePerShare', 'Details']
            st.write("Revenue/Share", revenuePerShare)
        except Exception:
            st.write("Data Unreported")
        try:
            returnOnAssets = dfi.at['returnOnAssets', 'Details']
            st.write("Return on Assets", returnOnAssets)
        except Exception:
            st.write("Data Unreported")
        try:
            returnOnEquity = dfi.at['returnOnEquity', 'Details']
            st.write("Return on Equity", returnOnEquity)
        except Exception:
            st.write("Data Unreported")
        try:
            pmargins = dfi.at['profitMargins', 'Details']
            st.write("Profit Margins", pmargins)  # 2b
        except Exception:
            st.write("Data Unreported")
        try:
            gmargins = dfi.at['grossMargins', 'Details']
            st.write("Gross Margins", gmargins)
        except Exception:
            st.write("Data Unreported")
        try:
            ebitdaMargins = dfi.at['ebitdaMargins', 'Details']
            st.write("EBITDA Margins", ebitdaMargins)
        except Exception:
            st.write("Data Unreported")
        try:
            operatingMargins = dfi.at['operatingMargins', 'Details']
            st.write("Operating Margins", operatingMargins)
        except Exception:
            st.write("Data Unreported")
        # st.write(dfi.at['TrailingPegRatio', 'Details'])
        try:
            grossProfits = dfi.at['grossProfits', 'Details']
            st.write("Gross Profits", grossProfits)
        except Exception:
            st.write("Data Unreported")
        try:
            earningsGrowth = dfi.at['earningsGrowth', 'Details']
            st.write("Earnings Growth", earningsGrowth)
        except Exception:
            st.write("Data Unreported")
        try:
            earningsQuarterlyGrowth = dfi.at['earningsQuarterlyGrowth', 'Details']
            st.write("Earnings Growth Qtrly", earningsQuarterlyGrowth)
        except Exception:
            st.write("Data Unreported")
        try:
            revenueGrowth = dfi.at['revenueGrowth', 'Details']
            st.write('Revenue Growth', revenueGrowth)
        except Exception:
            st.write("Data Unreported")
        try:
            payoutRatio = dfi.at['payoutRatio', 'Details']  # 5b
            st.write('Payout Ratio - ', payoutRatio)
        except Exception:
            st.write("Data Unreported")
        try:
            beta = dfi.at['beta', 'Details']
            st.write("Beta - ", beta)
        except Exception:
            st.write("Data Unreported")
        try:
            trailingPE = dfi.at['trailingPE', 'Details']
            st.write("Trailing PE", trailingPE)
        except Exception:
            st.write("Data Unreported")
        try:
            forwardPE = dfi.at['forwardPE', 'Details']
            st.write("Forward PE", forwardPE)
        except Exception:
            st.write("Data Unreported")
        try:
            bookValue = dfi.at['bookValue', 'Details']
            st.write("Book Value", bookValue)
        except Exception:
            st.write("Data Unreported")
        try:
            priceToBook = dfi.at['priceToBook', 'Details']
            st.write("Price-to-Book Ratio", priceToBook)
        except Exception:
            st.write("Data Unreported")
        try:
            netIncomeToCommon = dfi.at['netIncomeToCommon', 'Details']
            st.write("Net Income to Common", netIncomeToCommon)
        except Exception:
            st.write("Data Unreported")
        try:
            reco_mean = dfi.at['recommendationMean', 'Details']
            st.write('Recommended Mean - ', reco_mean)
        except Exception:
            st.write("Data Unreported")
        try:
            reco_key = dfi.at['recommendationKey', 'Details']
            st.write('Market Recommendation - ', reco_key)
        except Exception:
            st.write("Data Unreported")
        ep1, ep2 = st.columns(2)
        with ep1:
            try:
                trailingEps = dfi.at['trailingEps', 'Details']
                st.write("TrlgEPS", trailingEps)
            except Exception:
                st.write("Data Unreported")
            try:
                forwardEps = dfi.at['forwardEps', 'Details']
                st.write("FwdEPS", forwardEps)
            except Exception:
                st.write("Data Unreported")
        with ep2:
            try:
                enterpriseToRevenue = dfi.at['enterpriseToRevenue', 'Details']
                st.write("EV-Rev", enterpriseToRevenue)

            except Exception:
                st.write("Data Unreported")
            try:
                pegRatio = dfi.at['pegRatio', 'Details']
                st.write("PEG Ratio", pegRatio)
            except Exception:
                st.write("Data Unreported")
        st.write("  --------  ")
        # st.subheader("Institutional Risk Profile")

st.write("  --------------  ")
column1, column2, column3, column4, column5 = st.columns([1, 1, 1, 2, 1])
with column1:
    st.image(ytube, '[Ledgr\'s YouTube Channel](%s)' % url_ytube, width=60)
with column2:
    st.image(fbook, '[Ledgr\'s FaceBook Page ](%s)' % url_fb, width=60)
with column3:
    st.image(linkedin,  '[Our LinkedIn Page ](%s)' % url_linkedin, width=60)
with column4:
    st.write(" ")
    st.image(ledgrblog,  '[Ledgr\'s Blog ](%s)' % url_blog, width=85)
    st.write(" ")
with column5:
    st.image(insta,  '[Ledgr\'s @ Instagram ](%s)' % url_insta, width=60)
# # ###################################################################
with st.container():
    f9, f10, f11 = st.columns([1, 5, 1])
    with f9:
        st.write(" ")
    with f10:
        st.caption(
            ": | 2025 - 2026 | All Rights Resrved  ©  Ledgr Inc. | www.alphaLedgr.com | alphaLedgr Technologies Ltd. :")
    with f11:
        st.write(" ")
