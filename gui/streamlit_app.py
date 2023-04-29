import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import streamlit as st
import you
from auth import auth


def get_answer(question: str) -> str:
    # Set cloudflare clearance cookie and get answer from GPT-4 model
    try:
        result = you.Completion.create(prompt=question)

        return result["response"]

    except Exception as e:
        # Return error message if an exception occurs
        return (
            f'An error occurred: {e}. Please make sure you are using a valid cloudflare clearance token and user agent.'
        )

def init_app():
    # Set page configuration and add header
    st.set_page_config(
        page_title="gpt4freeGUI",
        initial_sidebar_state="expanded",
        page_icon="🧠",
        menu_items={
            'Get Help': 'https://github.com/xtekky/gpt4free/blob/main/README.md',
            'Report a bug': "https://github.com/xtekky/gpt4free/issues",
            'About': "### gptfree GUI",
        },
    )
    st.header('GPT4free GUI')

def run_app():
    # Add text area for user input and button to get answer
    question_text_area = st.text_area('🤖 Ask Any Question :', placeholder='Explain quantum computing in 50 words')
    if st.button('🧠 Think'):
        answer = get_answer(question_text_area)
        escaped = answer.encode('utf-8').decode('unicode-escape')
        # Display answer
        st.caption("Answer :")
        st.markdown(escaped)

    # Hide Streamlit footer
    hide_streamlit_style = """
                <style>
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


if __name__ == "__main__":
    init_app()

    if os.getenv("GPT4FREE_AUTH").lower() == "true":
        name, authentication_status, username, authenticator = auth()
        if authentication_status:
            authenticator.logout('Logout', 'main')
            st.write(f'Welcome *{name}*')
            run_app()
        elif authentication_status is False:
            st.error('Username/password is incorrect')
        elif authentication_status is None:
            st.warning('Please enter your username and password')
    else:
        run_app()
