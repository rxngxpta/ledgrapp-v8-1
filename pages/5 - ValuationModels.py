import numpy as np
import pandas as pd
# from pandas_datareader import data as pdr
import yfinance as yf
# from streamlit_pdf_viewer import pdf_viewer
import os
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import streamlit as st
import datetime as dt
# import base64
# import urllib
# Page Setup ##################################################################
sns.set()
plt.style.use('fivethirtyeight')
st.set_page_config(page_title='Ledgr | Valuation & Pricing Models',
                   layout="wide", initial_sidebar_state="expanded")
direc = os.getcwd()
# Declarations ################################################################
logofile = f'{direc}/pages/appdata/imgs/Ledgr_Logo_F2.png'

with st.sidebar:
    st.image(logofile, use_container_width=True)
    st.caption("Select a stock, train the algorithm and predict scenarios.")
# Icons and Links ###########################
ytube = f'{direc}/pages/appdata/imgs/ytube.svg'
fbook = f'{direc}/pages/appdata/imgs/fbook.svg'
insta = f'{direc}/pages/appdata/imgs/insta.svg'
linkedin = f'{direc}/pages/appdata/imgs/linkedin.svg'
ledgrblog = f'{direc}/pages/appdata/imgs/Ledgr_Logo_F1.png'
tickerfile = f"{direc}/pages/appdata/tickerlist_y.csv"
tickerdb = pd.read_csv(tickerfile)
tickerlist = tickerdb["SYMBOL"]
stock_m_list = ['^BSESN', '^NSEI']
periods = ['1y', '2y', '5y', '10y']
# Inputs #####################################################################
# Part I
start = dt.datetime(2019, 1, 1)
end = dt.datetime.now()  # ###################################################
mx1, mx2, mx3 = st.columns([2, 4, 2])
with mx1:
    st.write(' ')
with mx2:
    st.title(":ValuationModels:")
    st.write("**Find out how much an asset is worth or if a transaction is profitable compared to its peers.**")
with mx3:
    st.write(' ')


with st.form(key='Input Assset Info', clear_on_submit=False,
             enter_to_submit=True, border=True):
    stock = st.selectbox("Choose Stock Ticker", tickerlist)
    stock_m = st.selectbox('Choose the Base Index', stock_m_list)
    slider_val = st.slider("Expected Volatility")
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        stock_a = stock + ".NS"

        @st.cache_resource
        def CAPM(stock_a, stock_m):
            stock_a1 = yf.Ticker(stock_a)
            stock_m1 = yf.Ticker(stock_m)
            data_a = stock_a1.history(period='max')['Close']
            data_m = stock_m1.history(period='max')['Close']

            ME_stock_a = data_a.resample('ME').last()
            ME_stock_m = data_m.resample('ME').last()
            data = pd.DataFrame({'Inv_Close': ME_stock_a,
                                 'Markt_Close': ME_stock_m})
            data[['Inv_Ret', 'Markt_Ret']] = np.log(
                data[['Inv_Close',
                      'Markt_Close']]/data[['Inv_Close',
                                            'Markt_Close']].shift(1))
            data.dropna(inplace=True)
            beta_form = (data[['Inv_Ret',
                               'Markt_Ret']].cov()/data['Markt_Ret'].var()
                         ).iloc[0].iloc[1]
            beta_reg, alpha = np.polyfit(x=data['Markt_Ret'],
                                         y=data['Inv_Ret'], deg=1)
            st.write('\n')
            st.write(20*'==')
            st.metric('Calculated Beta - Linear Regression: ', round(beta_reg, 4))
            st.metric('Calculated Alpha: ', round(alpha, 4))
            st.write(20*'==')
            plt.figure(figsize=(13, 9))

            plt.axvline(0, color='grey', alpha=0.5)
            plt.axhline(0, color='grey', alpha=0.5)

            sns.scatterplot(y='Inv_Ret', x='Markt_Ret',
                            data=data, label='Returns')
            sns.lineplot(x=data['Markt_Ret'],
                         y=alpha + data['Markt_Ret']*beta_reg,
                         color='red', label='CAPM Line')

            plt.xlabel('Market Monthly Return: {}'.format(stock_m[0]))
            plt.ylabel('Investment Monthly Return: {}'.format(stock_a[0]))
            plt.legend(bbox_to_anchor=(1.01, 0.8), loc=2, borderaxespad=0.)
            st.pyplot(plt)

#        CAPM(stock_a, stock_m)

        @st.cache_resource
        def CAPM_daily(stock_a, stock_m):
            stock_a1 = yf.Ticker(stock_a)
            stock_m1 = yf.Ticker(stock_m)
            data_a = stock_a1.history(period='max')['Close']
            data_m = stock_m1.history(period='max')['Close']
            # ME_stock_a = data_a.resample('ME').last()
            # ME_stock_m = data_m.resample('ME').last()
            data = pd.DataFrame({'Inv_Close': data_a, 'Markt_Close': data_m})
            data[['Inv_Ret', 'Markt_Ret']] = np.log(
                data[['Inv_Close',
                      'Markt_Close']]/data[['Inv_Close',
                                            'Markt_Close']].shift(1))
            data.dropna(inplace=True)
            beta_form = (data[['Inv_Ret',
                               'Markt_Ret']].cov()/data['Markt_Ret'].var()
                         ).iloc[0].iloc[1]
            beta_reg, alpha = np.polyfit(x=data['Markt_Ret'],
                                         y=data['Inv_Ret'], deg=1)
            st.write('\n')
            st.write(20*'==')
            st.metric('Calculated Beta - Linear Regression: ', round(beta_reg, 6))
            st.metric('Calculated Alpha: ', round(alpha, 6))
            st.write(20*'==')
            plt.figure(figsize=(13, 9))
            plt.axvline(0, color='grey', alpha=0.5)
            plt.axhline(0, color='grey', alpha=0.5)

            sns.scatterplot(y='Inv_Ret',
                            x='Markt_Ret',
                            data=data, label='Returns')
            sns.lineplot(x=data['Markt_Ret'],
                         y=alpha + data['Markt_Ret']*beta_reg,
                         color='red', label='CAPM Line')

            plt.xlabel('Market Monthly Return: {}'.format(stock_m[0]))
            plt.ylabel('Investment Monthly Return: {}'.format(stock_a[0]))
            plt.legend(bbox_to_anchor=(1.01, 0.8), loc=2, borderaxespad=0.25)
            st.pyplot(plt)


#        CAPM_daily(stock_a, stock_m)
        pass
    if not submitted:
        st.stop()


st.header("Part1: Critical Asset Pricing Model")
v11, v12 = st.columns([1, 1])
with v11:
    st.subheader('1A. CAPM Plot: Monthly')
    CAPM(stock_a, stock_m)
with v12:
    st.subheader('1B. CAPM Plot: Daily')
    CAPM_daily(stock_a, stock_m)

st.write('------------------------------------------------------------------')
# Part II #####################################################################
ticker = yf.Ticker(stock_a)
stock_price = ticker.history(period='1y')['Close'][1]
# stock_price = stock_price.last()
st.write("Stock Price", stock_price)

info = ticker.info
info = pd.DataFrame([info])
info = info.rename(columns={0: 'Items', 1: 'Description'})
info.reset_index()
# info = info.set_index(['Items'])
pnl = ticker.financials
bsheet = ticker.balancesheet
cflow = ticker.cashflow
df_pnl = pd.DataFrame.from_dict(pnl)
df_cflow = pd.DataFrame.from_dict(cflow)
df_bsheet = pd.DataFrame.from_dict(bsheet)
##############################################################################
pnl2 = df_pnl.reset_index(level=None, drop=False, inplace=False, col_level=0)
cflow2 = df_cflow.reset_index(level=None, drop=False, inplace=False,
                              col_level=0, col_fill="")
bsheet2 = df_bsheet.reset_index(level=None, drop=False, inplace=False,
                                col_level=0, col_fill="")
pnl2 = pnl2.set_index('index')
bsheet2 = bsheet2.set_index('index')
cflow2 = cflow2.set_index('index')
###############################################################################
st.header("Part2: Firm's Financial Statements")
st.write('------------------------------------------------------------------')
t1, t2, t3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow Statement"])
with t1:
    st.caption(f'{stock}\'s Income Statement')
    st.dataframe(pnl2)
with t2:
    st.caption(f"{stock}\'s Balance Sheet")
    st.dataframe(bsheet2)
with t3:
    st.caption(f'{stock}\'s Cash Flow Statement')
    st.dataframe(cflow2)

url_ytube = "https://www.youtube.com/@LedgrInc"
url_fb = "https://www.facebook.com/share/1BnXaYvRzV/"
url_insta ='https://www.instagram.com/alphaledgr/'
url_blog = 'https://www.alphaledgr.com/Blog'
url_linkedin = "https://www.linkedin.com/company/ledgrapp/"
##############################################################################
st.write('------------------------------------------------------------------')
try:
    accounts_receivable = bsheet2.loc["Accounts Receivable"][0]
except Exception:
   accounts_receivable = 0


try:
    accounts_payable = bsheet2.loc["Accounts Payable"][0]
except Exception:
    accounts_payable = 0

try:
    inventory = bsheet2.loc["Inventory"][0]
except Exception:
    inventory = 1

try:
    taxes_payable = bsheet2.loc["Total Tax Payable"][0]
except Exception:
    taxes_payable = 'In the Works'

try:
    current_liabilities = bsheet2.loc["Current Liabilities"][0]
except Exception:
    current_liabilities = bsheet2.loc['Current Liabilities'][1]
else:
    pass

try:
    cost_of_rev = pnl2.iloc["Cost of Revenue"]
except Exception:
    cost_of_rev = "Insufficient Data"

try:
    cash = bsheet2.loc["Cash Financial"][0]
except Exception:
    cash = cflow2.loc["End Cash Position"][1]

try:
   current_assets =   bsheet2.loc['Current Assets'][0]
# mkt_price_per_share =
except Exception:
   current_assets = cash + accounts_receivable + inventory


try:
    book_value = bsheet2.loc['Tangible Book Value'][0]

except Exception:
    book_value = 'Insufficient Data'
try:
    equity = bsheet2.loc['Stockholders Equity'][0]

except Exception:
    equity = 'Data Unreported'


try:
    shares_outstanding = bsheet2.loc['Share Issued'][0]
except Exception:
    shares_outstanding = 'Data Unreported'
try:
    book_value_per_share = book_value/shares_outstanding
except Exception:
    book_value_per_share = 1

try:
    cash_flow_from_operations = cflow2.loc['Operating Cash Flow'][0]
except Exception:
    cash_flow_from_operations = st.warning("In the works")

try:
    OPEX = pnl2.loc['Operating Expense'][0]
except Exception:
    OPEX = "In the works"

try:
    sga = pnl2.loc['Selling General And Administration'][0]
# COGS =
except Exception:
    sga = "Data Unreported"
try:
    total_rev = pnl2.loc['Total Revenue'][0]
except Exception:
    total_rev = "Data Unreported"

try:
    net_income = pnl2.loc['Net Income'][0]
except Exception:
    net_income = "Data Unreported"
try:
    net_revenues = pnl2.loc['Normalized EBITDA'][0]
except Exception:
    net_revenues = "Data Unreported"

try:
    op_rev = pnl2.loc['Operating Revenue'][0]
except Exception:
    op_rev = "Data Unreported"

try:
    total_assets = pnl2.loc['Operating Revenue'][0]
except Exception:
    total_assets = pnl2.loc['Operating Revenue'][1]
else:
    total_assets = "Data Unreported"

try:
    shareholders_equity = bsheet2.loc['Stockholders Equity'][0]
except Exception:
    shareholders_equity = "Data Unreported"
try:
    cash_eqv = bsheet2.loc['Cash And Cash Equivalents'][0]
except Exception:
    cash_eqv = "Data Unreported"


try:
   eps = pnl2.loc['Basic EPC'][0]
except Exception:
   eps = 'Data Unreported'
   
try:
    cash_and_cash_eqv = bsheet2.loc['Cash Cash Equivalents & Short Term Investments'][0]
except Exception:
#    cash_and_cash_eqv = bsheet2.loc['Cash Cash Equivalents & Short Term Investments'][1]
# else:
    cash_and_cash_eqv = "Data Unreported"

try:
    working_capital = bsheet2.loc['Working Capital'][0]
except Exception:
    working_capital = "Data Unreported"

try:
    gross_profit = pnl2.loc['Gross Profit'][0]
except Exception:
    gross_profit = "Data Unreported"
try:
    total_debt = bsheet2.loc['Total Debt'][0]
except Exception:
    total_debt = "Data Unreported"

try:
    total_liabilities = bsheet2.loc['Total Liabilities Net Minority Interest'][0]
except Exception:
    total_liabilities = "Data Unreported"


try:
    interest_expenses = pnl2.loc['Interest Expense Non Operating'][0]
except Exception:
    interest_expenses = "Data Unreported"

try:
    ebitda = pnl2.loc['EBITDA'][0]
except Exception:
    ebitda = "Data Unreported"

try:
    current_debt = bsheet2.loc['Current Debt'][0]
except Exception:
    current_debt = "Data Unreported"

try:
    ebit = pnl2.loc["EBIT"][0]
except Exception:
    ebit = "Data Unreported"

try:
    cost_of_revenue = pnl2.loc['Cost Of Revenue'][0]
except Exception:
    cost_of_revenue = "Data Unreported"


try:
    dividends = bsheet2.loc['Dividends Payable'][0]
except Exception:
    dividends = "Data Unreported"

try:
    net_sales = total_rev - cost_of_revenue
except Exception:
    net_sales = "Data Insufficient"
try:
    op_profits = op_rev - OPEX
except Exception:
    op_profits = "Data Unreported"
try:
    net_profit = 0.7*gross_profit
except Exception:
    net_profit = "Data Unreported"

u1, u2, u3, u4, u5 = st.columns([1, 1, 1, 1, 1])
with u1:
    st.metric("Net Income (Cr.)", net_income/10000000)
with u2:
    st.metric("Current Assets (Cr.)", current_assets/10000000)
with u3:
    st.metric("Operating Profits (Cr.)", op_profits/10000000)
with u4:
    st.metric("Operating Cash Flow", cash_flow_from_operations/10000000)
with u5:
    st.metric("Gross Profit in (Cr.)", gross_profit/10000000)

# Pagework 2
st.write('-------------------------------------------------------------------')
st.title("Part 3: Financial Metrics")
st.write('-------------------------------------------------------------------')
st.header("1. Liquidity ratios.")
st.info('''Compute the availability of a company’s total underlying short-term
        assets, such as account receivables, to cover current liabilities,
        such as account payables.''')
st.write('-------------------------------------------------------------------')
st.subheader("1A. Current-Ratio")
st.info("""It tells investors and analysts how a company can maximize the
        current assets on its balance sheet to satisfy its current debt and
        other payables.
        A current ratio that is in line with the industry average or slightly
        higher is generally considered acceptable.""")


try:
    cr = current_assets/current_liabilities
except Exception:
    cr = st.warning("Data Unreported")



g2, g3 = st.columns(2)
with g2:
    st.metric("Current Assets in Cr.", current_assets/10000000)
with g3:
    st.metric("CR = (Current Assets) ÷ (Current Liabilities)", round(cr, 4))
st.write("""A Firm with a CR less than one means that its Equity heavy and
         shall not be able to manage its short term payouts""")
st.subheader("1B. Acid-Test-Ratio")
st.info('''The acid-test ratio (or quick ratio) is calculated by dividing a
        company\'s quick assets by its current liabilities.''')
try:
    acr = (cash + accounts_receivable)/current_liabilities
except Exception:
    acr = 'Probable Data Insufficiency'


g4, g5 = st.columns(2)
with g4:
    st.metric("""=>ACR=Quick assets(cash+accounts_rcv)/current liabilities""",
              round(acr, 4))
with g5:
    st.write("""A Firm with a CR less than one means that its Equity heavy and
             shall not be able to manage its short term payouts""")
st.subheader("1C. Cash-Ratio")
st.info('''Benchmark being around the value of 1, a higher Cash Ratio indicates
        that the firm is in good health, being able to deal with short and
        long term payments''')

try:
    c_ratio1 = (cash + cash_eqv)/current_liabilities
except Exception:
    c_ratio1 = 'In the Works'
try:
    c_ratio2 = current_assets/current_liabilities
except Exception:
    c_ratio2 = 'In the Works'
g6, g7 = st.columns(2)
with g6:
    st.info("**=>Cash Ratio=(Cash + Cash Equivalents)÷Current Liabilities)")
with g7:
    st.metric("CR = (Current Assets)÷(Current Liabilities)",
               round(c_ratio2, 4))
st.write("""If a Firm's cash ratio is around 0.6, it means that for every
         dollar borrowed, it has only 0.6 dollars to pay back.""")

try:
    ocfr = cash_flow_from_operations/current_liabilities
except Exception:
    ocfr = 'In the Works'
try:
    cogs = total_rev - gross_profit
except Exception:
    cogs = 'In the Works'
st.subheader("1D. Operating-Cash-Flow-Ratio")
g8, g9 = st.columns(2)
with g8:
    st.info('Measure how much a company earns against the amount borrowed.')
with g9:
    st.metric("**=>OCFR = Total Revenue - (COGS -+OPEX)**", round(ocfr, 4))
st.write("=>OCFR = Cash Flow from Operations/Current Liabilities")

st.write('-------------------------------------------------------------------')
#################################################
st.header("2. Leverage or Debt Ratios.")
st.info('''Leverage ratios help fathom whether a firm positioned in terms of
        its short-term and long-term obligations.''')
st.write('-------------------------------------------------------------------')
g10, g11 = st.columns(2)
with g10:
    st.metric("Total Debt in Cr.", total_debt/10000000)
with g11:
    st.info('''Measure the level of debt the company takes on to finance its
            operations, against the level of capital, or available equity''')
st.write('-------------------------------------------------------------------')
try:
    der = total_liabilities/shareholders_equity
    st.subheader("2.1 Debt-to-Equity-Ratio")
    g12, g13 = st.columns(2)
    with g12:
        st.info('''The debt-to-equity ratio indicates how much portion of the
                capital is borrowed and how much is invested in equity.''')
    st.write("""=>Debt-to-Equity:=(Total Liabilities)÷(Shareholders Equity)""")
    st.write('''If a company has a higher debt-to-equity ratio,
                 it means it is leveraging more,
                 and it is more vulnerable to interest rates.''')

    with g13:
        st.metric("Debt-to-Equity Ratio", round(der, 4))
except Exception:
    der = 'In the Works'

st.write('-------------------------------------------------------------------')
st.subheader("2.2 Interest-Coverage-Ratio")
try:
    icr = ebit/interest_expenses
    icr = icr.round(4)
except Exception:
    icr = 'In the Works'
h14, h15 = st.columns(2)
with h14:
    st.info('Gauge how easily a firm can pay interest on its outstanding debt')
    st.write('A High ICR i.e. above 2.00 exhibits a firm\'s capacity to pay off all ineterests of short term debts')
with h15:
    st.metric("Interest Coverage Ratio", icr)
try:
    dscr = ebit/current_debt
    dscr = dscr.round(4)
except Exception:
    dscr = 'In the Works'
st.subheader("2.3 Debt-Service-Coverage-Ratio")
g14, g15 = st.columns(2)
with g14:
    st.info('''Measures a firm\'s available cash flow to pay
            its current debt obligations''')
with g15:
    st.metric("Debt-Service-Coverage-Ratio", dscr)
st.write('-------------------------------------------------------------------')
# ################################################
st.header("3. Efficiency ratios.")
st.info("""An efficiency ratio measures a company’s ability to use its assets
        to generate income. It often looks at various aspects of the company,
        such as the time it takes to collect cash from customers or,
        to convert inventory to cash.
        Improvements in efficiency ratio translates to profitability.""")
st.write('-------------------------------------------------------------------')
try:
    atr = net_sales/total_rev
except Exception:
    atr = 'In the Works'
st.subheader("3.1 Asset-Turnover-Ratio")
g16, g17 = st.columns(2)
with g16:
    st.info('Measure how a company uses its assets to generate revenue.')
with g17:
    st.metric("**=> ATR = Net Sales/Total Revenues**", round(atr, 4))
try:
    itr = cogs/inventory
except Exception:
    itr = 'In the Works'
st.subheader("3.2 Inventory-Turnover-Ratio")
h17, h18 = st.columns(2)
with h17:
    st.info('''Inventory turnover is the rate that inventory stock is sold,
            or used, and replaced''')
with h18:
    st.metric('**=>COGS/Avg. Value of Inventory**', round(itr, 4))

st.write('''Measures how many times a Firm\'s Inventory is used and
         sold over a period''')

st.write('-------------------------------------------------------------------')
try:
    dsi = (inventory*365)/cogs
except Exception:
    dsi = 'In the Works'
st.subheader("3.3 Day-Sales-in-Inventory-Ratio")
g18, g19 = st.columns(2)
with g18:
    st.info('Mark the avg days it takes to turn inventory into sales')
with g19:
    st.metric("=>DSI = (Avg. Value of Inventory/COGS)X365: DSI", round(dsi, 4))
st.write('-------------------------------------------------------------------')
# ################################################
st.header("4. Profitability ratios.")
st.info('''Measure and evaluate the ability of a company to generate
        income (profit) relative to revenue, balance sheet assets,
        operating costs, and shareholders’ equity during a specific period.
        They show how well a company utilizes its assets to produce profit and
        value to shareholders.''')
st.write('-------------------------------------------------------------------')
try:
    gmr = (total_rev - cogs)/net_revenues
except Exception:
    gmr = 'In the Works'
st.subheader("4.1 Gross-Margin-Ratio")
g20, g21 = st.columns(2)
with g20:
    st.info('''The gross margin ratio is a percentage that shows how much gains
            a company makes for each dollar of revenue.''')
with g21:
    st.metric("**=>GMR = (Revenues - COGS)/Revenues**", round(gmr, 4))
st.write("""A higher ratio indicates that a company is efficiently managing
         its costs and pricing""")

try:
    omr = 100*op_profits/op_rev
except Exception:
    omr = 'In the Works'
st.subheader("4.2 Operating-Margin-Ratio")
g22, g23 = st.columns(2)
with g22:
    st.info('Measure a company\'s operating profit relative to its revenues')
with g23:
    st.metric("**=>OPR = Operating Profits*100/Net Revenues", round(omr, 4))
try:
    roe = net_profit/shareholders_equity
except Exception:
    roe = 'In the Works'
st.subheader("4.3 Return-on-Equity")
g24, g25 = st.columns(2)
with g24:
    st.info('Informs you of the company’s net profit w,r.t its equity capital')
with g25:
    st.metric("ROE = (Net Profit)÷(Avg Shareholder’s Equity)", round(roe, 4))
    # roe = net_profit/Average_Shareholders_Equity
st.write('-------------------------------------------------------------------')
st.header("5. Market value ratios. ")
st.info("""Measure, analyze and compare stock prices against market prices,
        competitors accounting in other relevant writes.""")
st.write("""Mark out the optimal price levels
         at which a security's stock should be bought or sold. """)
st.write("-------------------------------------------------------------------")
st.subheader('5.1 Book-Value')
g42, g43 = st.columns(2)
with g42:
    st.info("Gauge if the stock is worth buying.")
with g43:
    st.metric("""=>BV=Total Stockholders' Equity/(Total Outstanding Shares)""",
             round(book_value/10000000, 4))
st.write("""This is the valuation as reflected in the audited books
         of the company.""")

st.subheader('5.2 Price-to-Book-Ratio')
bvps = book_value/shares_outstanding
pbr = stock_price/bvps
#pbr = round(pbr, 3)
g28, g29 = st.columns(2)
with g28:
    st.info("""Accounts in the firm's market value,
            book value and Shares Outstanding. Firms with Low P-BV,
            in certain scenarios, indicate a 'Buy',
            given that one has considered other relevant factors.""")
with g29:
    st.metric("P-B Ratio:", round(pbr, 4))
st.write("=>P-BV = (Market Price Per Share)÷(Book Value per share)")
try:
   p_s = (stock_price*shares_outstanding)/total_rev
except Exception:
   p_s ='Insufficient Data'

st.subheader('5.3 Price-to-Sales-Ratio')
h1, h2 = st.columns(2)
with h1:
    st.info('Gauge a security\'s price vs its Sales Turnover')
    st.caption("P/S = (Price per Share) ÷ (Annual Sales Per Share)")
with h2:
    st.metric("Price-to-Sales", round(p_s, 6))

st.subheader('5.4 Market-to-Book-Ratio')
try:
    bvps = book_value/shares_outstanding
    mbr = stock_price/bvps
    mbr = round(mbr, 6)
except Exception:
    mbr = 'In the Works'
g32, g33 = st.columns(2)
with g32:
    st.info('Market Value here refers to the stock price')
with g33:
    st.metric("Market-to-Book Ratio:", mbr)

st.subheader('5.5 Dividend-Yield')
g34, g35 = st.columns(2)
with g34:
    st.info('''Estimate how much dividends
            the company has declared to its Stockholders''')
with g35:
    st.caption('=>Div Yield=(Annual Dividend per Share)÷(Price per Share)×100')
try:
    ceps = net_income/shares_outstanding
    ceps = round(ceps, 3)
except Exception:
    ceps = 'Invalid Outputs'
st.subheader('5.6 Earnings-per-Share')
g36, g37 = st.columns(2)
with g36:
    st.info("EPS represents the earnings on every unit of stock you own.")
with g37:
    st.metric("=>Calculated EPS = [Net Income]/(Total Outstanding Shares)",
             ceps)
st.write("=>EPS = [Net Profit-Preferred_Shares_Dividend]/(Shares_Outstanding)")

try:
    dps = dividends/shares_outstanding
    dps = round(dps, 3)
except Exception:
    dps = 'In the Works'
st.subheader('5.7 Dividend-per-Share')
g38, g39 = st.columns(2)
with g38:
    st.info('''Accounts for the sum of declared dividends issued by a company
            for every share outstanding.''')
with g39:
    st.metric("Dividend-per-Share:",dps)
pe = shares_outstanding*stock_price/net_income
pe = round(pe, 3)
st.subheader('5.8 Price-to-Earnings-Ratio')
g40, g41 = st.columns(2)
with g40:
    st.info("Gauge if the stock is Over/Undervalued")
with g41:
    st.metric("=>Calculated PE=(Price Per Share)÷(Earnings Per Share)", pe)
st.write("A PE ratio of 42 means that investors are willing to pay for every ₹1 of the company's earnings per share(EPS)")

st.subheader('5.9 Dividend-Payout-Ratio')
try:
    dpr = (dividends/net_income)
    dpr = round(dpr, 3)
except Exception:
    dpr = 'Data Inconsistent'
g44, g45 = st.columns(2)
with g44:
    st.info('''(DPR) is the amount of dividends paid to shareholders
            accounting in the total net income the company generates''')
with g45:
    st.metric("**=>DPR = Total Dividends/Net Income**", dpr)
st.write("**=>DPR = Dividends per share/Earnings per share**")
st.write('------------------------------------------------------------------')
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

