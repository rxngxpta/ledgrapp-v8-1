# !/usr/bin/env python310
# -*- coding: utf-8 -*-
# __author__ = 'R. Sengupta | r_xn'
# __copyright__ = 'Copyright 2023, Ledgr | www.alphaLedgr.com'
# __credits__ = ['r_xn, s.sengupta, adasgupta@gmail.com']
# __license__ = 'Ledgr | alphaledgr.com'
# __version__ = '01.02.04'
# __maintainer__ = 'r_xn@alphaledgr.com'
# __emails__ = 'r_xn@alphaledgr.com / response@alphaledgr.com'
# __status__ = 'In active development'

# Imports #####################################################################
import numpy as np
import pandas as pd
import streamlit as st
import base64
import os
import urllib
# from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer

st.set_page_config(page_title='Ledgr | About & Tutorials', layout="wide",
                   initial_sidebar_state="expanded")

direc = os.getcwd()
logofile = f'{direc}/pages/appdata/imgs/Ledgr_Logo_F2.png'

url_ytube = "https://www.youtube.com/@LedgrInc"
url_fb = "https://www.facebook.com/share/1BnXaYvRzV/"
url_insta ='https://www.instagram.com/alphaledgr/'
url_blog = 'https://www.alphaledgr.com/Blog'
url_linkedin = "https://www.linkedin.com/company/ledgrapp/"
with st.sidebar:
    st.image(logofile, use_container_width=True)
    st.caption("The How-to's, Docs, Demos and your Queries")
# Icons and Links ###########################
ytube = f'{direc}/pages/appdata/imgs/ytube.svg'
fbook = f'{direc}/pages/appdata/imgs/fbook.svg'
insta = f'{direc}/pages/appdata/imgs/insta.svg'
linkedin = f'{direc}/pages/appdata/imgs/linkedin.svg'
ledgrblog = f'{direc}/pages/appdata/imgs/Ledgr_Logo_F1.png'


n1, n2, n3 = st.columns([1, 5, 1])
with n1:
    st.write(" ")
with n2:
    st.title("About & Documentation")
with n3:
    st.write(" ")
n12, n13, n14 = st.columns([2, 4, 2])
with n12:
    st.write(" ")
with n13:
    st.header("How does Ledgr work")
with n14:
    st.write(" ")
st.write("  ------  ")
with st.container():
    j1, j2, j3 = st.columns([1, 6, 1])
    with j1:
        st.write(' ')
    with j2:
        st.video('https://youtu.be/0i0MQqovGb8?si=nda1SJazcrfvTFUc')
        j11, j12, j13 = st.columns([3, 3, 2])
        with j11:
            st.write(" ")
        with j12:
            st.write("Ledgr - An Overview")
        with j13:
            st.write(" ")
        st.write(20*"---")

        st.video('https://youtu.be/FQyN4dc5Myo?si=nS_mUOyGDshVDfaJ')
        k11, k12, k13 = st.columns([2, 4, 2])
        with k11:
            st.write(" ")
        with k12:
            st.link_button("Ledgr's Business Model Canvas", "https://www.alphaledgr.com/blog")
            st.write(" ")

    with j3:
        st.write(' ')

st.write("-------------------------------------------------------------------")
m1, m2 = st. columns(2)
with m1:
    st.header("Navigate using the Sidebar!!")
    st.markdown(
        """
    - Each app has its unique functionality and presents unique information.

    - Find out more by clicking on them!! Choose any via the Sidebar > .

    - On each page -

        (a) input relevant details,

        (b) authorize the access fee payment

        (c) click submit!!. That's it.

    """)
with m2:
    st.header("Each module is a vending machine!!")
    st.markdown(
        """
    - Assess information, data, plots and signals.

    - Gather overviews, access in-depth analyses on markets & instruments.

    - Get a comprehensive picture of your total wealth,
    \n along with in-depth insights as you navigate through the app-engines.

    - Use LedgrTokens to activate engines and modules as you would like.
    \n Analyze, assess and access the data as a download file.

    """)

st.write("  ---------------------------------------------------------------  ")
uu2, uu3 = st.columns([3, 5])
with uu2:
    st.subheader(":LedgrBase:")
    st.caption("Map your investment portfolios here!")
    st.markdown("""
        - Organize all your expenses at the LedgrBase, which your homepage.
        - Unify your asset holdings here.
        - Visualize everything in a set of convenient, interactive dashboards.
        - Build a new Portfolio [For new Users], save it to your profile or share.
        - Track each portfolio at the LedgrBase.
        - Add other asset holdings to 'Your Locker'.
        - Asset classes accepted are currently Securities.
        - Derivatives, Bonds, ETFs & MFs, Crypto & Fiat Money are in development.
        - Provisions for other classes viz Derivatives etc will be included.
        - Trace Performance, Returns due, and gain a clear summary of your Wealth,
                """)
with uu3:
    img12 = f"{direc}/pages/appdata/imgs/Users-Assets.png"
    st.image(img12, use_container_width=True)
    # vf1 = open(f"{direc}/pages/appdata/imgs/LP1.mp4", "rb")
    # vb1 = vf1.read()
    # st.video(vb1, format="mp4", start_time=0,
    #          loop=True, autoplay=False, muted=False)
st.write("  ---------------------------------------------------------------  ")
vv23, vv33 = st.columns([3, 5])
with vv23:
    st.image(f"{direc}/pages/appdata/imgs/MarketBoard1.png")
with vv33:
    st.subheader(":MarketBoard:")
    st.caption("Follow Markets, Explore Funds & SIPs and more... ")
    st.markdown(
        """
            - Follow Markets, trace Market Indices.
            - Track their performances over time along with other Markets & Exchanges.
            - Visualize their comparative performance on the opening chart.
            - Navigate through the tabs for specific economies/indices.
            - Explore ETFs and MFs. Get quotes & information summaries.
            - Calculate SIP Returns on the SIP Calculator.
            - Find out your next Returns by inputting relevant data.
            - Know about Performers, Sectoral Activities and commodities.
            - Data from Crypto-currencies like BTC, ETH etc. demand inclusion.
            - Get info on Derivatives like Futures and Option Chains.
            - Get exchange rates and track Volatility indices like India VIX.
            - Know Treasury Rates to gauge the Markets condition holistically.
                """)
st.write("  ---------------------------------------------------------------  ")
va23, va33 = st.columns([3, 5])
with va23:
    st.image(f"{direc}/pages/appdata/imgs/Analytics1.png")
with va33:
    st.subheader(":AnalyticsBox:")
    st.markdown(
        """
        - Get OHLC Price Charts, Volume plots and all relevant info.
        - Access 42+ technical indicators, peruse stochastic signals.
        - Perform complex analyses on Securities with reported data.
        - Generate data reports and download calculated data.
        - Follow Price Movements and other KPIs.
        - Gain insights essential to make informed decisions and ensure maximum returns for your trades or investments.
        """)
    st.image(f"{direc}/pages/appdata/imgs/ANALYZING-AN-ASSET.png",
              use_container_width=False)

st.write("  ---------------------------------------------------------------  ")
vg23, vg33 = st.columns([3, 5])
with vg23:
    st.image(f"{direc}/pages/appdata/imgs/InvestLab1.png")
with vg33:
    st.subheader(":InvestmentLab:")
    st.caption("Optimize Portfolios, generate efficient allocations. Enjoy greater ROI.")
    st.markdown(
        """
            - Optimize your Investments! Start with a new perspective.
            - Input Securities to include in your Portfolio.
            - Indicate the total amount that you would be willing to allocate !!
            - Drop in the access fees and click submit !!
            - Users with Cashcards have their token automatically deducted from their connected LedgrWallets.*
            - InvestmentLab presents 5 different sets of optimally allocated portfolios with expected outcomes as per your inputs. Choose any one and invest effectively using the InvestmentLab.
            - Alternatively, one may select allocations to try out any other combinations of stocks and explore possibilities!
        """)

st.image(f"{direc}/pages/appdata/imgs/Reasons-for-using-the-portfolio-optimizer.png", use_container_width=True)

st.write("  ---------------------------------------------------------------  ")
vv23, vv33 = st.columns([3, 5])
with vv23:
    st.image(f"{direc}/pages/appdata/imgs/Forecast1.png")
with vv33:
    st.subheader(":ForecastEngine:")
    st.caption("Train ML-AI Engines, Predict Prices and gather intelligence")
    st.markdown("""
            - Predict future price ranges for any asset or security with Ledgr's AI ForecastingEngines.
            - Get Price Forecasts & Sentimental Analyses on specific securites, overall Markets or specific Market Segments!! Assess yearly, monthly as well as weekly motion of the prices.
            - Explore how security prices move through selected timespan.
            - Run it yourself, by your own rules. Use Ledgr's LSTM, ARIMA or any one of Ledgr's AI Models, input information and then adjust parameters of the engine suiting your requirement, prior to running the algorithms.
            - Please note that on your instruction, the AI model will execute in real-time.
            *Hence, sometimes it may seem to be taking long or the browser may have stalled. However, in reality the AI is running in the back-end.*
            - On completion, predicted prices are presented along with a set of other information.
              """)
    st.image(f"{direc}/pages/appdata/imgs/Forecasting-an-Asset.png")

st.write("  ---------------------------------------------------------------  ")

st.subheader("DiscussionRow")
st.write("In current Design & Development")
st.markdown(
        '''
            - Discuss about your Portfolio, or your Wealth Journey at DiscussionRow forums.

            - Share your Portfolios and or holdings, share opinions, observations, knowledge and most importantly, memes and degeneracy.

            - Content is organized by topics and threads, Users interact via Likes/Dislikes, Comments, shares and posts.

            - Users can build profiles, have friends and interact via the global "DiscussionWall" or across threads, groups, etc.

            - Users can earn LedgrTokens in a few ways other than purchasing them from the LedgrExchange portal.

            They are -

                (a) as initial rewards, intermittent grants and random AirDrops

                (b) by interacting on the platform and performing specific tasks [KYC, Referrals, Subscriptions to sub-plans etc.]

                (c) by participating in collab-work protocols, due-diligence-dropoffs for eg.

                    - Threads are moderated, and some moderators shall be selected from active Users, to work along with experts.

                    - Get great curated info, news, links, updates etc along with a variety of other content and resources.

                    - Build the community, get engaged, grow together.
            ''')


ytube = f'{direc}/pages/appdata/imgs/ytube.svg'
fbook = f'{direc}/pages/appdata/imgs/fbook.svg'
insta = f'{direc}/pages/appdata/imgs/insta.svg'
linkedin = f'{direc}/pages/appdata/imgs/linkedin.svg'
ledgrblog = f'{direc}/pages/appdata/imgs/Ledgr_Logo_F1.png'
dpiit = f'{direc}/pages/appdata/imgs/AlphaLedgr Certificate of Recognition Startup India.png'
# #############################################################################
# nx1, nx2, nx3 = st.columns([3, 2, 3])
# with nx1:
#     st.write(" ")
# with nx2:
#     st.title(":FAQ, Links etc:")
# with nx3:
#     st.write(" ")
# nn12, nn13, nn14 = st.columns([3, 2, 3])
# with nn12:
#     st.write(" ")
# with nn13:
#     st.write("Find FAQs, Documentation & Slide Decks etc.")
#     st.write("Get Visuals, Videos and of course, grab the LedgrMerch here!!")
# with nn14:
#     st.write(" ")
st.write(" ------------------------------------------------------------------------------------------------------------------ ")

# ###############################
st.header("2. Frequently Asked Questions - FAQs and Queries")
st.caption("FAQs and Queries")
with st.container():
    s1, s2 = st.columns(2)
    with s1:

        st.write("**a. How does Ledgr look to integrate with the existing players?**")
        st.markdown("Ledgr's Value and Service Offerings are different, so are Ledgr's principles of User-driven outlook.")

        st.markdown("Ledgr seeks to integrate and strengthen Users and Markets and therefore seeks to not cannibalize on the Market Capitalization of any Firm in the same sector. The reason is rather simple. Ledgr seeks to ensure Efficient, Knowledge based Decision making processes of Users. The onus of implimenting the insights are on the User. Ledgr's job is to present a lucid, transparent, customizable platform to Users as explained in our Blog Section.")
        st.write("**b. Does one have to know hardcore finance to use Ledgr?**")
        st.markdown("The short answer is No. For cases otherwise, worry not, Ledgr has a dedicated module and shall develop wikis on every component with references.")
        st.markdown("Ledgr aims to make people aware, thereby enabling them to make their independant decisions.")
        st.write("c. **How does one make sustainable wealth with Ledgr?**")
        st.markdown("One can find detailed tutorials and 'How-Tos' in our 'Blog', or the 'About and Tutorials' pages.")
        st.write("**d. What is the Ledgr and how do I access it?**")
        st.markdown(""" Ledgr is an integral of 3 entities. 1. The alphaLedgr.com
                   website has all the relevant information, signup and
                   access links and also hosts the blog, documentation,
                   contacts etc. LedgrApp is a Web3.0 platform which
                   is trust independent. This demands a network which is secure
                   by design aka the LedgrExchange and a transaction framework
                   which hosts both fiat and digital transactions,
                   with a Digital Wallet System aka the LedgrWallet.
                   They are integrated, the records are immutable to promote
                   transparency while being future-proof, auditable by authorities,
                   is easy to use and is globally scalable.""")
    with s2:
        st.markdown("""The Website hosts the LedgrApp Link on the Homepage.
                   The LedgrExchange and the LedgrWallet also have Signup and
                   Registration Links. Ledgr's overall integration makes the
                   User experience seamless, engaging, and collaborative.""")
        st.write("**e. Where can one find the Product Details and Manuals?**")
        st.markdown("""There are dedicated sections for each in both the 'Blog'
                   and the 'App' """)
        st.write("**f. How can one accrue wealth using the InvestmentLab?**")
        st.markdown("""This is a venture into the InvestmentLab.
                    There's a tutorial video for creating a portfolio""")
        st.write("**g. Are LedgrTokens cryptocurrency?**")
        st.markdown("""No. Blockchains are frameworks and tokens are components
                    which make the Firm, its Engines and their auxilliaries
                    enable seamless in-house operations primarily.
                    There are security measures involved as per standard protocol,
                    but no transparency issues.
                    Ledgr Tokens, if the situation calls for it,
                    can and may be audited but only by Authorized Personnel
                    from the Reserve Bank or the Regulatory Authorities
                    exclusively, for ensuring transparent audits
                    and Official Reporting. Ledgr also shall feature a
                    Subscription Model. The User is free to choose.""")
        st.write("**h. How do I know that my money is safe with Ledgr?**")
        st.markdown(""" Ledgr operates on Blockchain Utility Tokens,
                    just akin to the ones we used in the banks.
                    These tokens are immutable, thereby ensuring the validity
                    and security of every transaction. Tokens can be exchanged
                    against fiat-money at the LedgrExchange Portal and get
                    stored in your LedgrWallet. Whatever the invested amount be,
                    it is going to be insured to protect all parties. And secondly,
                    the Tokens, are purchased as per the Users' requirement as
                    Ledgr believes in reducing redundant costs
                    to the Users themselves.""")

st.subheader("1.3. Resources & Links")
ss2, ss3, ss4 = st.columns(3)
with ss2:
    st.image(f'{direc}/pages/appdata/imgs/Ledgr_Logo_F2.png', "https://alphaledgr.com/blog")
    st.markdown("alphaledgr.com's dedicated blog and Ledgr's Whitepaper")
with ss3:
    st.write("The LedgrExchange Portal")
    st.link_button("Go to The LedgrExchange Portal", "https://alphaLedgr.com/")
with ss4:
    st.write("The LedgrWallet System")
    st.link_button("Go to The LedgrExchange Portal", "https://alphaLedgr.com/")
###############################################################################
file0A = f'{direc}/pages/appdata/imgs/Ledgr_Whitepaper_and_Business_Monograph.pdf'
file0 = f'{direc}/pages/appdata/imgs/AlphaLedgr Certificate of Recognition Startup India.pdf'
file1 = f'{direc}/pages/appdata/imgs/Udyam_Registration_Certificate_Final.pdf'

st.header("2. Documentation")
with st.container():
    st.subheader("2.1. Docs and Manuals")
    st.write("2.1.1. Value Proposition, Vision, and Service Proposal")
    st.video("https://youtu.be/G6QGCUchP0k?si=w1WWv9snNdHbZenC")
    gl1, gl2, gl3 = st.columns([2, 4, 2])
    with gl1:
        st.write(" ")
    with gl2:
        st.link_button("Value Proposition, Vision, and Service Proposal",
                       "https://www.alphaledgr.com/blog/ledgr-the-firm/business-model")
    with gl3:
        st.write(' ')
    ff1, ff2, ff3 = st.columns([1, 4, 1])
    with ff1:
        st.write(' ')
    with ff2:
        st.video("https://youtu.be/FQyN4dc5Myo")
        st.write("2.1.2. Product Offering")
    with ff3:
        st.write(' ')
    l1, l2, l3 = st.columns([2, 3, 2])
    with l1:
        st.write(" ")
    with l2:
        st.image(f'{direc}/pages/appdata/imgs/product_architecture.png', use_container_width=True)
        st.link_button("B. Ledgr's Product Architecture",
                       "https://www.alphaledgr.com/blog/how-ledgr-works/app-architecture")

        st.link_button("C.  Blog & Documentation", "https://www.alphaledgr.com/blog/ledgr-the-firm/blog")
    with l3:
        st.write(" ")

with st.container():
    st.subheader("2.2. Recognition & Certifications")


    d1, d2 = st.columns(2)
    with d1:
        st.write("**2.2.1. DPIIT Recognition**")
        with open(file0, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display0 = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="350" height="250" type="application/pdf">'
        st.markdown(pdf_display0, unsafe_allow_html=True)
    st.write(" --------- ")
    with d2:
        st.write("**2.2.2. UDYAM - MSME Certificate**")
        with open(file1, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="350" height="250" type="application/pdf">'
        st.markdown(pdf_display, unsafe_allow_html=True)

##############################################################################


st.write(" ------------------------------------------------------------ ")
st.header("3. Ledgr's Partners and Affiliates")
s1, s2, s3 = st.columns(3)
with s1:
    st.subheader("Our Merch Partners")
    st.image(f'{direc}/pages/appdata/imgs/shady.png' ,"https://shadystuffs.com", width=75)
    st.write("Check them out here!")
with s2:
    st.subheader("Our Tech Partners")
    # st.image(f'{direc}/pages/appdata/imgs/shady.png' ,"https://shadystuffs.com", width=75)
    st.write(" ")
with s3:
    st.subheader("Our Financial Partners")
    # st.image(f'{direc}/pages/appdata/imgs/shady.png' ,"https://shadystuffs.com", width=75)
    st.write(" ")

st.write(" --------------------------------------------------------- ")
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

