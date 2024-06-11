import streamlit as st
import requests
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from hugchat import hugchat

st.set_page_config(page_title="Visualization")

with st.sidebar:
    st.title('Visualization')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LlamaIndex](x)
    - [OpenAI](x) LLM model
    
    💡 Note: No API key required!
    ''')
    add_vertical_space(5)
    st.write('Made by Sulaiman')

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm X, How may I help you?"]
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

def get_text() -> str:
    """
    Get user input from text input box.
    
    Returns:
        str: The user input.
    """
    input_text = st.text_input("You: ", "", key="input")
    return input_text

with input_container:
    user_input = get_text()

backend_url = "http://localhost:8000/chat"  # Assuming both are running on localhost (Can be adjusted)

def generate_response(user_input: str) -> str:
    """
    Generate a response from the backend based on user input.
    
    Args:
        user_input (str): The user input.

    Returns:
        str: The generated response from the backend.
    """
    response = requests.post(backend_url, json={"user_input": user_input})
    return response.json()["bot_response"]

with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
