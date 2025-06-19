# !/usr/bin/env python310
# -*- coding: utf-8 -*-
# __author__ = 'R. Sengupta | r_xn'
# __copyright__ = 'Copyright 2023, Ledgr | www.alphaLedgr.com'
# __credits__ = ['r_xn, s.sengupta,  adasgupta@gmail.com']
# __license__ = 'Ledgr | alphaledgr.com'
# __version__ = '01.02.04'
# __maintainer__ = 'r_xn@alphaledgr.com'
# __emails__ = 'r_xn@alphaledgr.com / response@alphaledgr.com'
# __status__ = 'In active development'
#  streamlit_authenticator as stauth

from yaml.loader import SafeLoader
import os
import streamlit as st

st.set_page_config(page_title="Home | Ledgr", page_icon=None,
                   layout="centered", initial_sidebar_state="expanded")
direc = os.getcwd()

##################################################################
logofile = f'{direc}/pages/appdata/imgs/Ledgr_Logo_F2.png'

with st.sidebar:
    st.image(logofile)
    st.caption("Your unified Fintelligence Portal!")

# #######################################
# Icons and Links ###########################
ytube = f'{direc}/pages/appdata/imgs/ytube.svg'
fbook = f'{direc}/pages/appdata/imgs/fbook.svg'
insta = f'{direc}/pages/appdata/imgs/insta.svg'
linkedin = f'{direc}/pages/appdata/imgs/linkedin.svg'
ledgrblog = f'{direc}/pages/appdata/imgs/Ledgr_Logo_F2.png'
icon_size = 100

url_ytube = "https://www.youtube.com/@LedgrInc"
url_fbook = "https://www.facebook.com/share/1BnXaYvRzV/"
url_insta ='https://www.instagram.com/alphaledgr/'
url_blog = 'https://www.alphaledgr.com/Blog'
url_linkedin = "https://www.linkedin.com/company/ledgrapp/"
# Main Streamlit app starts here

st.markdown(''' <div align="center"><h1>Hello! Welcome to Ledgr.</h1></div>''',
            unsafe_allow_html=True)
st.markdown('''
            <div align="center">
            <h3>Learn how to get started on the platform!
            See below for details..</h3></div>''',
            unsafe_allow_html=True)

st.image(f'{direc}/pages/appdata/imgs/The alphaLedgr Web3 Platform.png',
         use_container_width=True)
with st.container():
    a1, a2a, a2, a3 = st.columns([1, 1, 4, 1])
    with a1:
        st.image(f'{direc}/pages/appdata/imgs/LedgrBase.svg',
                 caption='Your Unified Wealth Dashboard')
    with a2a:
        st.write(" ")
    with a2:
        st.subheader("Part I: Ledgrbase")
        st.write("Map your existing asset holdings and portfolios.")
        st.write("Review and note their overall performance till date.")
        st.subheader("Part II: MarketBoard")
        st.write("Calculate Returns from SIPs, Explore ETFs and Mutual Funds.")
    with a3:
        st.image(f'{direc}/pages/appdata/imgs/MarketBoard.png',
                 caption='Market Profiles, Plots and Instruments')
st.write("-------------------------------------------------------------------")
with st.container():
    c1, c2a, c2 = st.columns([1, 1, 3])
    with c1:
        st.image(f'{direc}/pages/appdata/imgs/AnalyticsBox.png',
                 caption='Analytics and Information')
    with c2a:
        st.write(" ")
    with c2:
        st.subheader("AnalyticsBox")
        st.write("Analyze a Security In-Depth. Visualize Metrics & Indicators")
        st.write("Gather comprehensive knowhow on a selected Security.")
st.write("-------------------------------------------------------------------")
with st.container():
    d1, d2a, d2 = st.columns([3, 1, 1])
    with d1:
        st.subheader("InvestmentLab")
        st.write("Optimize Investment Allocations.")
        st.write("Generate Efficient Portfolios to Maximize Returns.")
        st.write("Input assets and amount to proceed. Select any from 5 Optimized portfolios presented.")

    with d2a:
        st.write(" ")
    with d2:
        st.image(f'{direc}/pages/appdata/imgs/InvestmentLab.png',
                 caption='Generate Optimal Portfolios', use_container_width=True)
st.write("-------------------------------------------------------------------")
with st.container():
    e1, e2a, e2 = st.columns([1, 1, 3])

    with e1:
        st.image(f'{direc}/pages/appdata/imgs/ForecastEngine.png',
                 'Forecast Price Ranges using your own inputs.')
    with e2a:
        st.write(" ")
    with e2:
        st.subheader("ForecastEngine")
        st.write("Train Ledgr's AI")
        st.write("Generate price forecasts for any securities or currencies. Observe overall trend plots aover multiple timescales")
        st.write("Use the docs and the posts on our Blog to commence! Please do not forget to drop in your feedback")
st.write("-------------------------------------------------------------------")
column1, column2, column3, column4, column5 = st.columns([1, 1, 1, 2 , 1])
with column1:
    st.image(ytube, '[Ledgr\'s YouTube Channel](%s)' % url_ytube, width=60)
with column2:
    st.image(fbook, '[Ledgr\'s FaceBook Page ](%s)' % url_fbook, width=60)
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
        st.caption(": | 2025 - 2026 | All Rights Resrved  ©  Ledgr Inc. | www.alphaLedgr.com | alphaLedgr Technologies Ltd. :")
    with f11:
        st.write(" ")

