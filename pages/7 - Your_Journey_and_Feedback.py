import pandas as pd
# import datetime as dt
import streamlit as st
import os

st.set_page_config(page_title='Ledgr | Contact Us', layout="wide",
                   initial_sidebar_state="expanded")


direc = os.getcwd()
st.header("Contact, Suggestions & Communication")
logofile = f'{direc}/pages/appdata/imgs/Ledgr_Logo_F2.png'
with st.sidebar:
    st.image(logofile, use_container_width=True)
    st.caption("Some Feedback Please! Guide us, help us grow!")
    st.write('Support the project!')

# #######################################
# Icons and Links ###########################
ytube = f'{direc}/pages/appdata/imgs/ytube.svg'
fbook = f'{direc}/pages/appdata/imgs/fbook.svg'
insta = f'{direc}/pages/appdata/imgs/insta.svg'
linkedin =f'{direc}/pages/appdata/imgs/linkedin.svg'
ledgrblog =f'{direc}/pages/appdata/imgs/Ledgr_Logo_F1.png'

url_ytube = "https://www.youtube.com/@LedgrInc"
url_fb = "https://www.facebook.com/share/1BnXaYvRzV/"
url_insta ='https://www.instagram.com/alphaledgr/'
url_blog = 'https://www.alphaledgr.com/Blog'
url_linkedin = "https://www.linkedin.com/company/ledgrapp/"
icon_size = 100
st.sidebar.caption("Hi. Thanks for your time with Ledgr. Ledgr develops on active guidance from its Users and Visitors. Any suggestion and feedback is welcome!")

st.write("  ---------------------------------------------------------------  ")
c21, c22 = st.columns(2)
with c21:
    st.header("Drop in some wisdom here, please!")
    st.write("Please let us know about your experience and suggestions below:")
    with st.form('Feedback'):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        phone = st.number_input("Your Contact Number [Not Mandatory]")
        feedbck_1 = st.text_area("Please write in your message here!!")
        feedbck_2 = st.text_area("Any additional features/ideas/suggestions?")
        exp_level = st.slider(label='Rate your Experience out of 10!',
                              min_value=1, max_value=10,
                              value=6, step=1, help=None)
        submitted = st.form_submit_button("Submit")
        if submitted:
            df_feedback_2 = pd.DataFrame({"Name": [name], "Email": [email],
                                          "Phone": [phone],
                                          "feedbck_1": [feedbck_1],
                                          "feedbck_2": [feedbck_2],
                                          "exp_level": [exp_level]})
            st.write(df_feedback_2)
            df_feedback_2.to_csv(f"{direc}/appdata/User_FBack.csv")
st.write("  ---------------------------------------------------------------  ")
with c22:
    st.header("2. Acknowledgements")
    st.subheader("*Mentors, Critiques, Collaborators and other partners in crime!*")
    st.caption("To you guys, LedgrTeam fail to articulate their gratitude. Best we can do is deliver, with your guidance as our rudder.")
    st.caption('The list is inexhaustive. People help us by the day. We shall do our best to include as many of them here!')
    list1 = ["Dr. A & S Dasguptas", "Mr. G Bhattacharyya", " Mr. Debayan Chaterjee", "Mr. Pritam Saha", " Mr. D Bose", "Mr. T Sengupta", "Mrs U Sen", "Mr M Sarkar", "Mrs. R Davidson", "Mr M. A. Thiriat", 'Mr Owen Poulain', "Dr. Arnab Dasgupta", "Dr. D. Dasgupta"]
    list2 = ["Guidance, Teaching and Support","Trust & Tactical Advice", "Tactical Advice", "Tactical Advice", "Adoptee and Support", "Support", "Teaching, Advice and Support", "Technical Advisory, Training and Tactical knowhow", "Inspiration and Mentorship", "Support, Inspiration and Guidance",'Partner in Crime', "Hard Guidance, Advisory and observatorship", "Observatorship & Moderating"]

    st.header("The Ledgr Community")
    contri1 = pd.DataFrame({"Name or CallSign": list1,
                            'Contribution': list2})
    contri1

st.write("  ---------------------------------------------------------------  ")
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

