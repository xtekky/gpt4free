from gpt4free import you
import streamlit as st
import os
import sys
from typing import Union

# Add parent directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

class GPT4FreeModel:
    def __init__(self):
        self.model = you.Completion

    def get_answer(self, question: str) -> Union[str, None]:
        try:
            result = self.model.create(prompt=question)
            return result.text
        except Exception as e:
            return None, e

def display_error_message(e: Exception) -> str:
    return (
        f'An error occurred: {e}. Please make sure you are using a valid cloudflare clearance token and user agent.'
    )

def main():
    # Set up Streamlit page configuration
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
    
    # Display the header
    st.header('GPT4free GUI')

    # Add a text area for users to input their question
    question_text_area = st.text_area(
        '🤖 Ask Any Question :', placeholder='Explain quantum computing in 50 words')
    
    gpt4free_model = GPT4FreeModel()

    # Process the question when the "Think" button is clicked
    if st.button('🧠 Think'):
        answer, error = gpt4free_model.get_answer(question_text_area)
        
        if answer:
            escaped = answer.encode('utf-8').decode('unicode-escape')
            st.caption("Answer :")
            st.markdown(escaped)
        else:
            st.error(display_error_message(error))

    # Hide Streamlit footer
    hide_streamlit_style = """
                <style>
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
