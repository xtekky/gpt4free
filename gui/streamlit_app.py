import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import streamlit as st
import phind

# Set cloudflare clearance and user agent
phind.cloudflare_clearance = ''
phind.phind_api = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'


def get_answer(question: str) -> str:
    # Set cloudflare clearance cookie and get answer from GPT-4 model
    try:
        result = phind.Completion.create(
            model='gpt-4',
            prompt=question,
            results=phind.Search.create(question, actualSearch=True),
            creative=False,
            detailed=False,
            codeContext=''
        )
        return result.completion.choices[0].text
    except Exception as e:
        # Return error message if an exception occurs
        return f'An error occurred: {e}. Please make sure you are using a valid cloudflare clearance token and user agent.'


# Set page configuration and add header
st.set_page_config(
    page_title="gpt4freeGUI",
    initial_sidebar_state="expanded",
    page_icon="🧠",
    menu_items={
        'Get Help': 'https://github.com/xtekky/gpt4free/blob/main/README.md',
        'Report a bug': "https://github.com/xtekky/gpt4free/issues",
        'About': "### gptfree GUI"
    }
)
st.header('GPT4free GUI')

# Add text area for user input and button to get answer
question_text_area = st.text_area(
    '🤖 Ask Any Question :', placeholder='Explain quantum computing in 50 words')
if st.button('🧠 Think'):
    answer = get_answer(question_text_area)
    # Display answer
    st.caption("Answer :")
    st.markdown(answer)

# Hide Streamlit footer
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
