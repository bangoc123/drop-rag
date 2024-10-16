import streamlit as st


@st.dialog("Notification")
def notify(msg: str="", noti_type: str=""):
    if noti_type == "error":
        st.error(msg)
    elif noti_type == "warning":
        st.warning(msg)
    else:
        st.info(msg)