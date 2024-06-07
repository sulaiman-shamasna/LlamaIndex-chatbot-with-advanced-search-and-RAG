import streamlit as st
import requests

st.title("Chatbot Interface")

backend_url = "http://backend:8000/chat"

def get_response(user_input):
    response = requests.post(backend_url, json={"user_input": user_input})
    return response.json()["bot_response"]

user_input = st.text_input("You: ", "")

if st.button("Send"):
    if user_input:
        bot_response = get_response(user_input)
        st.text_area("Bot:", value=bot_response, height=200)
