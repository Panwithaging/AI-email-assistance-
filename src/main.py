import streamlit as st

from classify import classify
from preprocess import preprocessing
from summarize import summarizer
from action import require_action
from reply import reply

st.set_page_config(page_title="Email assistant V0",
                   layout="wide",
                   initial_sidebar_state="expanded")
st.title("THIS IS A EMAIL ASSISTANT VERSION 0 ")
st.header("this model is very slow it might take minutes to generate")
mail=st.text_area(label="Paste your email",
             height=300,
             placeholder="paste ur whole email")
if st.button("process"):
    st.session_state["mail"]=preprocessing(mail)
    st.session_state["Notification"]=summarizer(st.session_state["mail"])
    st.session_state["actions"]=require_action(st.session_state["mail"])
    

if "Notification" in st.session_state:
    st.subheader("Notification")
    st.write(st.session_state["Notification"])
    if "actions" in st.session_state and len(st.session_state["actions"])>0:
        select=st.selectbox("select",st.session_state["actions"])
        if select=="Reschedule":
            date=st.date_input("date",format="DD-MM-YYYY")
            time=st.time_input("time")
        if st.button("Generate reply"):
            if select=="Reschedule":
                st.session_state["Reply"]=reply(st.session_state["mail"],select,date,time)
            else:
                st.session_state["Reply"]=reply(st.session_state["mail"],select)


    if "Reply" in st.session_state:
        st.subheader("Reply")
        st.write(st.session_state["Reply"])


