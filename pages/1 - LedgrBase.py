# __author__ = 'R. Sengupta | r_xn'
# __copyright__ = 'Copyright 2023, Ledgr | www.alphaLedgr.com'
# __credits__ = ['r_xn, s.sengupta, prithvirajsengupta033@gmail.com]
# __license__ = 'Ledgr | alphaledgr.com'
# __version__ = '01.02.04'
# __maintainer__ = 'r_xn@alphaledgr.com'
# __emails__ = 'r_xn@alphaledgr.com / response@alphaledgr.com'
# __status__ = 'In active development'


# import numpy as np
import pandas as pd
import datetime as dt
import plotly.express as px
import yfinance as yf
# from nsepy import get_history, get_index_pe_history
import plotly.graph_objs as go
import os
# import pickle
# import matplotlib as plt
# import seaborn as sns
from plotly.subplots import make_subplots
from mftool import Mftool
from selectolax.parser import HTMLParser
import requests
import streamlit as st

st.set_page_config(page_title='LedgrBase | Your Asset Dossier',
                   layout="wide", initial_sidebar_state="expanded")
# ##################################################################

direc = os.getcwd()
# st.write(direc)
# direc = f'{direc}/Documents/Ledgr'
logofile = f'{direc}/pages/appdata/imgs/Ledgr_Logo_F2.png'
st.logo(logofile, size="medium", link='https://alphaledgr.com/Blog/',
        icon_image=logofile)
with st.sidebar:
    st.image(logofile)
    st.caption("View Markets, get info on funds & monitor your Holdings!")
mf = Mftool()
start_date = dt.datetime(2021, 1, 1)
end_date = dt.datetime.today()
altstart = dt.datetime(2023, 1, 1)
indlist = pd.read_csv(f'{direc}/pages/appdata/Index_L.csv')['Symbol']
indlist = pd.Series(indlist)
etflist = pd.read_csv(f'{direc}/pages/appdata/ETF_L.csv')['Symbol']
tickerl = pd.read_csv(f'{direc}/pages/appdata/tickerlist_y.csv')['SYMBOL']
mflist = pd.read_csv(f'{direc}/pages/appdata/mfcodes.csv')
curr_list = pd.read_csv(f'{direc}/pages/appdata/currency_list.csv')['Symbol']
userdf = pd.read_csv(f'{direc}/pages/appdata/userloggedin.csv')
# ######################################################################
username_list = userdf['UserLoggedIN']
username = username_list.iloc[-1]
usertag = userdf['Usertag']
usertag = usertag.iloc[-1]
tstamp = dt.datetime.now()
userdf2 = pd.DataFrame({'UserLoggedIN': [username],
                        'Usertag': [usertag], 'Timestamp': [tstamp]})
# userdf2 = userdf2.set_index(['Timestamp'], inplace=True)
userdf = pd.concat([userdf, userdf2])
# userdf2.to_csv('userloggedin.csv', index=False)
st.title("Your Wealth Dashboard and Global Finances")
# ####################################################
# Icons and Links ###########################
ytube = f'{direc}/pages/appdata/imgs/ytube.svg'
fbook = f'{direc}/pages/appdata/imgs/fbook.svg'
insta = f'{direc}/pages/appdata/imgs/insta.svg'
linkedin = f'{direc}/pages/appdata/imgs/linkedin.svg'
ledgrblog = f'{direc}/pages/appdata/imgs/Ledgr_Logo_F1.png'
icon_size = 100  # ####################################

hh1, hh2 = st.columns([2, 3])
with hh1:
    st.header(": LedgrBase :")
    st.subheader('Hi User!')
    st.subheader("Welcome to Ledgr!")
    st.write('Organize your asset-holdings here, track their performance!')

with hh2:
    st.video("https://youtu.be/m8C4C-LW3YY")

# #####################################################
m1, m2, m3 = st.columns([3, 4, 3])
with m1:
    st.write(" ")
with m2:
    st.markdown("""This section is under works and shall be released with
                the main release of the LedgrApp.""")
with m3:
    st.write(" ")

# ##############################################################################


@st.cache_resource
def data_BSE():
    BSE = yf.Ticker('^BSESN')
    df_BSE = BSE.history(period='5y')
    figOHLC_BSE = go.Figure()
    figOHLC_BSE.add_trace(go.Ohlc(x=df_BSE.index,
                                  open=df_BSE["Open"],
                                  high=df_BSE["High"],
                                  low=df_BSE["Low"],
                                  close=df_BSE["Close"],
                                  name="SENSEX"))
    figOHLC_BSE.update_layout(xaxis_rangeslider_visible=False)
    return df_BSE, figOHLC_BSE


df_BSE, figOHLC_BSE = data_BSE()


@st.cache_resource
def data_NSEI():
    nse = yf.Ticker('^NSEI')
    df_NSEI = nse.history(period='5y')
    figOHLC_NSEI = go.Figure()
    figOHLC_NSEI.add_trace(go.Ohlc(x=df_NSEI.index,
                                   open=df_NSEI["Open"],
                                   high=df_NSEI["High"],
                                   low=df_NSEI["Low"],
                                   close=df_NSEI["Close"],
                                   name="NIFTY50"))
    figOHLC_NSEI.update_layout(xaxis_rangeslider_visible=False)
    return df_NSEI, figOHLC_NSEI


df_NSEI, figOHLC_NSEI = data_NSEI()


@st.cache_resource
def data_SPX():
    spx = yf.Ticker('^GSPC')
    df_SPX = spx.history(period='5y')
    figOHLC_SPX = go.Figure()
    figOHLC_SPX.add_trace(go.Ohlc(x=df_SPX.index, open=df_SPX["Open"],
                                  high=df_SPX["High"],
                                  low=df_SPX["Low"], close=df_SPX["Close"],
                                  name="SPX"))
    figOHLC_SPX.update_layout(xaxis_rangeslider_visible=False)
    return df_SPX, figOHLC_SPX


df_SPX, figOHLC_SPX = data_SPX()


@st.cache_resource
def data_DAX():
    dax = yf.Ticker('^GDAXI')
    df_DAX = dax.history(period='5y')
    figOHLC_DAX = go.Figure()
    figOHLC_DAX.add_trace(go.Ohlc(x=df_DAX.index, open=df_DAX["Open"],
                                  high=df_DAX["High"],
                                  low=df_DAX["Low"],
                                  close=df_DAX["Close"],
                                  name="DAX"))
    figOHLC_DAX.update_layout(xaxis_rangeslider_visible=False)
    return df_DAX, figOHLC_DAX


df_DAX, figOHLC_DAX = data_DAX()


@st.cache_resource
def data_CAC():
    cac = yf.Ticker("^FCHI")
    df_CAC = cac.history(period='5y')
    figOHLC_CAC = go.Figure()
    figOHLC_CAC.add_trace(go.Ohlc(x=df_CAC.index,
                                  open=df_CAC["Open"],
                                  high=df_CAC["High"],
                                  low=df_CAC["Low"],
                                  close=df_CAC["Close"], name="CAC40"))
    figOHLC_CAC.update_layout(xaxis_rangeslider_visible=False)
    return df_CAC, figOHLC_CAC


df_CAC, figOHLC_CAC = data_CAC()


@st.cache_resource
def data_DJIA():
    dji = yf.Ticker("^DJI")
    df_DJIA = dji.history(period='5y')
    figOHLC_DJIA = go.Figure()
    figOHLC_DJIA.add_trace(go.Ohlc(x=df_DJIA.index,
                                   open=df_DJIA["Open"], high=df_DJIA["High"],
                                   low=df_DJIA["Low"],
                                   close=df_DJIA["Close"],
                                   name="DJIA"))
    figOHLC_DJIA.update_layout(xaxis_rangeslider_visible=False)
    return df_DJIA, figOHLC_DJIA


df_DJIA, figOHLC_DJIA = data_DJIA()


@st.cache_resource
def data_TYO():
    tyo = yf.Ticker("^N225")
    df_tyo = tyo.history(period='5y')
    figOHLC_tyo = go.Figure()
    figOHLC_tyo.add_trace(go.Ohlc(x=df_tyo.index,
                                  open=df_tyo["Open"], high=df_tyo["High"],
                                  low=df_tyo["Low"], close=df_tyo["Close"],
                                  name="TYO"))
    figOHLC_tyo.update_layout(xaxis_rangeslider_visible=False)
    return df_tyo, figOHLC_tyo


df_tyo, figOHLC_tyo = data_TYO()


@st.cache_resource
def data_FTSE():
    FTSE = yf.Ticker("^FTSE")
    df_FTSE = FTSE.history(period='5y')
    figOHLC_FTSE = go.Figure()
    figOHLC_FTSE.add_trace(go.Ohlc(x=df_FTSE.index,
                                   open=df_FTSE["Open"],
                                   high=df_FTSE["High"],
                                   low=df_FTSE["Low"],
                                   close=df_FTSE["Close"],
                                   name="FTSE"))
    figOHLC_FTSE.update_layout(xaxis_rangeslider_visible=False)
    return df_FTSE, figOHLC_FTSE


df_FTSE, figOHLC_FTSE = data_FTSE()


@st.cache_resource
def data_mkt():
    df_mk = pd.DataFrame()
    df_mk['SENSEX'] = df_BSE['Close']
    df_mk['NSEI'] = df_NSEI['Close']
    df_mk['DAX'] = df_DAX['Close']
    df_mk['CAC'] = df_CAC['Close']
    df_mk['SPX'] = df_SPX['Close']
    df_mk['FTSE'] = df_FTSE['Close']
    df_mk['N225'] = df_tyo['Close']
    fig_mkt = go.Figure()
    fig_mkt.add_trace(go.Scatter(x=df_NSEI.index, y=df_NSEI['Close'],
                                 mode='lines', name='NSEI'))
    fig_mkt.add_trace(go.Scatter(x=df_BSE.index, y=df_BSE['Close'],
                                 mode='lines', name='SENSEX'))
    fig_mkt.add_trace(go.Scatter(x=df_DAX.index, y=df_DAX['Close'],
                                 mode='lines', name='DAX'))
    fig_mkt.add_trace(go.Scatter(x=df_CAC.index, y=df_CAC['Close'],
                                 mode='lines', name='CAC'))
    fig_mkt.add_trace(go.Scatter(x=df_SPX.index, y=df_SPX['Close'],
                                 mode='lines', name='SPX'))
    fig_mkt.add_trace(go.Scatter(x=df_DJIA.index, y=df_DJIA['Close'],
                                 mode='lines', name='DJIA'))
    fig_mkt.add_trace(go.Scatter(x=df_FTSE.index, y=df_FTSE['Close'],
                                 mode='lines', name='FTSE'))
    fig_mkt.add_trace(go.Scatter(x=df_tyo.index, y=df_tyo['Close'],
                                 mode='lines', name='N225'))
    fig_mkt.update_xaxes(visible=True, showticklabels=True)
    fig_mkt.update_yaxes(visible=True, showticklabels=True)

    return df_mk, fig_mkt


df_mk, fig_mkt = data_mkt()
multi_symbols = ['^IXIC',
                 '^GSPC', '^NYA',
                 '^BSESN',
                 '^NSEI', '^NSEBANK']
multi_details = ['NASDAQ Composite', 'S&P500', 'NYSE Composite (DJ)',
                 'BSE SENSEX',  'NIFTY50',
                 'NIFTYBANK']

multi_index_list = pd.DataFrame({"Symbol": multi_symbols,
                                 "Exchange Index": multi_details})

# ^TYX - US Treasury Yield - 30 years


@st.cache_resource
def treasury():
    trs = yf.Ticker('^TYX')
    df_treasury = trs.history(period='5y')
    fig_treasury = go.Figure()
    fig_treasury.add_trace(go.Ohlc(x=df_treasury.index,
                                   open=df_treasury["Open"],
                                   high=df_treasury["High"],
                                   low=df_treasury["Low"],
                                   close=df_treasury["Close"]))
    fig_treasury.update_xaxes(visible=True, showticklabels=True)
    fig_treasury.update_yaxes(title='US Treasury Yield', visible=True,
                              showticklabels=True)
    fig_treasury.update_layout(xaxis_rangeslider_visible=False,
                               showlegend=False)
    return df_treasury, fig_treasury


df_treasury, fig_treasury = treasury()


@st.cache_resource
def vix():
    vix = yf.Ticker('^VIX')
    df_vix = vix.history(period='5y')
    # df_vix = df_vix.drop(['Volume'], axis=1)
    fig_vix = go.Figure()
    fig_vix.add_trace(go.Ohlc(x=df_vix.index, open=df_vix["Open"],
                              high=df_vix["High"],
                              low=df_vix["Low"], close=df_vix["Close"],
                              name="VIX"))
    fig_vix.update_traces(increasing_line_color='yellow',
                          decreasing_line_color='gray')
    fig_vix.update_layout(xaxis_rangeslider_visible=False)
    fig_vix.update_xaxes(visible=True, showticklabels=True)
    fig_vix.update_yaxes(title='VIX', visible=True, showticklabels=True)
    fig_vix.update_layout(height=360, showlegend=False)
    return df_vix, fig_vix


df_vix, fig_vix = vix()

mf = Mftool()
# you can use as_json=True to get all codes in json format
all_scheme_codes = mf.get_scheme_codes()

st.write("    ----    ")
hg1, hg2 = st.columns([2, 3])
with hg1:
    st.header(': MarketBoard :')
    st.subheader('Follow, Track and Global Markets')
    st.caption('Explore Indices, Exchange Traded & Mutual Funds and more!')
with hg2:
    st.video("https://youtu.be/it2owsaWp5c")

st.subheader('A. Markets & Exchanges')
st.info(""" Compare Global Markets. Investigate each Markets performance in the tabs which follow""")
tub0, tub1, tub1A, tub2, tub3, tub4, tub5, tub6, tub7 = st.tabs(['Global Markets',
                                                          "NSE - IN",
                                                          "BSE - SENSEX",
                                                          "SPX - USA",
                                                          "DAX - GDR",
                                                          'CAC40 - FR',
                                                          'Dow Jones - US',
                                                          'Nikkei225 - JPN',
                                                          "FTSE - UK"])
with tub0:
    st.plotly_chart(fig_mkt, use_container_width=True)
with tub1:
    st.plotly_chart(figOHLC_NSEI, use_container_width=True)
with tub1A:
    st.plotly_chart(figOHLC_BSE, use_container_width=True)
with tub2:
    df_SPX, figOHLC_SPX = data_SPX()
    st.plotly_chart(figOHLC_SPX, use_container_width=True)
with tub3:
    st.plotly_chart(figOHLC_DAX, use_container_width=True)
with tub4:
    st.plotly_chart(figOHLC_CAC, use_container_width=True)
with tub5:
    st.plotly_chart(figOHLC_DJIA, use_container_width=True)
with tub6:
    st.plotly_chart(figOHLC_tyo, use_container_width=True)
with tub7:
    st.plotly_chart(figOHLC_FTSE, use_container_width=True)


st.write("  --------  ")


st.subheader("B. SIP Calculator")
st.caption("Find out your Returns from any SIP scheme.")

with st.container():
    with st.form('sipcalc'):
        A = st.slider("Enter the monthly SIP amount: ", min_value=500,
                      max_value=9900, value=1050, step=100,
                      help="Input your monthly payments installments here!")
        YR = st.slider("Enter the yearly Rate of Return in pct: ", min_value=5,
                       max_value=20, value=10, step=1,
                       help="Indicate your scheme's Return Rate[ref: IRR or XIRR]")
        Y = st.slider("Enter the number of years: ", min_value=2,
                      max_value=15, value=5, step=1,
                      help="Indicate the number of years of investing")
        submitted = st.form_submit_button("Calculate Returns >> ")
        if submitted:
            MR = YR/12/100
            M = Y * 12
            FV = A * ((((1 + MR)**(M))-1) * (1 + MR))/MR
            FV = round(FV)
            gh2, gh3 = st.columns(2)
            with gh2:
                st.subheader("Your Expected Returns are: - ")
            with gh3:
                st.metric("Returns [INR]", FV)
        else:
            st.warning("Select values and click Calculate Returns")
st.write("    ------    ")


@st.cache_resource
def df_snav(mf_sel):
    snav = mf.get_scheme_historical_nav(f"{mf_sel}")
    data = snav['data']
    return snav, data


@st.cache_resource
def etf(etfselect):
    etfselect = etfselect + '.NS'
    etf = yf.Ticker(etfselect)
    df_etf = etf.history(period='5y')
    figOHLC_etf = make_subplots(rows=2, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.2,
                                subplot_titles=('NAV Price Movement',
                                                'Traded Volume'),
                                row_width=[0.2, 0.7])
    figOHLC_etf.add_trace(go.Ohlc(x=df_etf.index,
                                  open=df_etf["Open"],
                                  high=df_etf["High"],
                                  low=df_etf["Low"],
                                  close=df_etf["Close"],
                                  name=f"OHLC for {etfselect}"),
                          row=1, col=1)
    figOHLC_etf.add_trace(go.Bar(
        x=df_etf.index, y=df_etf['Volume'], name='Volume Traded',
        showlegend=False), row=2, col=1)
    figOHLC_etf.update_layout(xaxis_rangeslider_visible=False)
    figOHLC_etf.update_layout(showlegend=False)
    return figOHLC_etf, df_etf


@st.cache_resource
def currency(currency_selected):
    curr = yf.Ticker(currency_selected)
    currency_df = curr.history(period='2y')
    currency_df1 = currency_df.filter(['Open', 'High', 'Low', 'Close'], axis=1)
    fig_currency1 = go.Figure()
    fig_currency1.add_trace(go.Ohlc(x=currency_df1.index,
                                    open=currency_df1["Open"],
                                    high=currency_df1["High"],
                                    low=currency_df1["Low"],
                                    close=currency_df1["Close"]))
    fig_currency1.update_xaxes(visible=True, showticklabels=True)
    fig_currency1.update_yaxes(title='Exchange Ratio', visible=True,
                               showticklabels=True)
    fig_currency1.update_layout(xaxis_rangeslider_visible=False, height=360)
    return currency_df1, fig_currency1


st.subheader('C. Mutual Funds')
st.caption('Map your Mutual Funds Here...')
df_scodes = pd.DataFrame(all_scheme_codes.items())
df_scodes.tail(4)
with st.form('mf_info'):
    mf_sel = st.selectbox("Choose Scheme Code", df_scodes)
    submitted = st.form_submit_button("Proceed")
    if not submitted:
        st.write("Select a Mutual Fund Code!!")
        pass
    if submitted:
        #    units_balance = st.number_input("Input your Units held in Balance"
        # , 500)
        q = mf.get_scheme_quote(mf_sel)
        df_q = pd.DataFrame(q.items())
        df_q.rename(columns={0: 'Items', 1: 'Details'}, inplace=True)
        df_q.set_index('Items')
        scheme_det = mf.get_scheme_details(mf_sel)
        scheme_details = pd.DataFrame(
            scheme_det.items(), columns=['Items', "Details"])
        scheme_details.tail()
        scheme_details = scheme_details.set_index('Items')
        scheme_details.sort_values('Items')
        fh = scheme_details.loc['fund_house'][0:]
        stype = scheme_details.loc['scheme_type'][0:]
        scat = scheme_details.loc['scheme_category'][0:]
        sname = scheme_details.loc['scheme_name'][0:]
        scode = scheme_details.loc['scheme_code'][0:]
        df_mk, fig_mkt = data_mkt()
        snav, data = df_snav(mf_sel)
        df_data = pd.DataFrame(data)
        df_data['date'] = pd.to_datetime(df_data['date'], format="%d-%m-%Y")
#       df_data.sort_values(by='date')
        st.write('1. Fund_house', fh.iloc[-1])
        st.write('2. Scheme Type', stype.iloc[-1])
        st.write('3. Scheme Category', scat.iloc[-1])
        st.write('4. Scheme Name', sname.iloc[-1])
        st.write('5. Scheme Code', scode.iloc[-1])
        st.write('6. Quotation', df_q)
        with st.expander("7. Expand here for NAV Data"):
            st.write(f"Security Code {mf_sel}", df_data)
        try:
            df_data.sort_values(by='date', ascending=True)
            f_mf = px.line(df_data, x=df_data['date'], y=df_data['nav'])
            st.plotly_chart(f_mf)
        except Exception:
            st.write("Needs more Work")
st.write("   ----   ")

url_ytube = "https://www.youtube.com/@LedgrInc"
url_fb = "https://www.facebook.com/share/1BnXaYvRzV/"
url_insta = 'https://www.instagram.com/alphaledgr/'
url_blog = 'https://www.alphaledgr.com/Blog'
url_linkedin = "https://www.linkedin.com/company/ledgrapp/"


st.subheader('D. Exchange Traded & Index Funds')
etfselect = st.selectbox("Please select ETF here!", etflist)
figOHLC_etf, df_etf = etf(etfselect)
st.plotly_chart(figOHLC_etf, use_container_width=True)
st.write("   ----   ")

st.header('E. Currencies')
currency_selected = st.selectbox("Select Currency Pair", curr_list)
currency_df1, fig_currency1 = currency(currency_selected)
cd1 = currency_df1['Close'].iloc[-1]
cd2 = 1/cd1
c11, c12 = st.columns([5, 1])
with c11:
    with st.expander("Get the data here!"):
        st.write(currency_df1)
with c12:
    st.metric("Exchange Rate:", cd2.round(2))
st.info("""Map Excange Rates across Currencies. The metric as presented above shows how much of the initial currency compensates for a unit of the following currency.""")
st.plotly_chart(fig_currency1)
st.write("  --------  ")
df_vix, fig_vix = vix()
l_vix = df_vix.iloc[-1]
cn1, cn2, cn3 = st.columns([3, 2, 1])
with cn1:
    st.header("F. Market Volatility Index")
with cn2:
    st.write(" ")
    st.markdown('''Estimate Uncertainty Levels in the Markets to gauge your Risk Exposure''')
with cn3:
    st.metric("Market VIX", l_vix['Close'].round(2))
st.plotly_chart(fig_vix, use_container_width=True)
st.write("  --------  ")
df_treasury, fig_treasury = treasury()
l_ustreasury = df_treasury['Close'].iloc[-1]

bn1, bn2, bn3 = st.columns([3, 2, 1])
with bn1:
    st.header("G. Treasury Yield Rates")
with bn2:
    st.write(" ")
    st.markdown('''Estimate the real Risk-Free rate >> Yield of the Treasury bond - Inflation Rate''')
with bn3:
    st.write(" ")
    st.metric("US Treasury", l_ustreasury.round(2))

tr1, tr2 = st.tabs(["US Treasury", "Reserve Bank of India"])
with tr1:
    st.plotly_chart(fig_treasury, use_container_width=True)
with tr2:
    user_agent = requests.get(
        "https://techfanetechnologies.github.io/latest-user-agent/user_agents.json").json()[-2]
    headers = {
        "user-agent": user_agent,
    }

    def riskFreeInterestRate(url: str = "https://www.rbi.org.in/",) -> None:
        response = HTMLParser(requests.get(url, headers=headers).content)
        selector = "#wrapper > div:nth-child(10) > table"
        data = [node.html for node in response.css(selector)]
        df = (pd.read_html(data[0])[0][4:13])
        df.columns = ["GovernmentSecurityName", "Percent"]
        df.reset_index(inplace=True, drop=True)
        df["GovernmentSecurityName"] = df["GovernmentSecurityName"].str.rstrip(
            ' ').str.lstrip(' ')
        df["Percent"] = df["Percent"].str.rstrip(
            '% #').str.rstrip('%*').str.lstrip(':  ')
#     df = df.astype({'GovernmentSecurityName': 'str',
    #                'Percent': 'float32'}, copy=False)
        with open("RiskFreeInterestRate.json", "w") as jsonFile:
            jsonFile.write(df.to_json(orient='records'))
        l1, l2 = st.columns([2, 3])
        with l1:
            st.write(df)
        df2 = df[0:8]
        with l2:
            fig_RFI2 = px.pie(df2, values='Percent', names='GovernmentSecurityName', title='Government Security Offerings')
            fig_RFI2.update_traces(textposition='outside', textinfo='percent+label')
            fig_RFI2.update_layout(showlegend=False)
            st.plotly_chart(fig_RFI2)

    if __name__ == "__main__":
        riskFreeInterestRate()

st.write("  --------  ")
c0, column1, column2, column3, column4, column5, c0a = st.columns([1, 1, 1, 1, 1, 1, 1])
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

