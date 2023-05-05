import os
import sys
import streamlit_lottie
import requests
import time

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import streamlit as st
from gpt4free import you


def get_answer(question: str) -> str:
    # Set cloudflare clearance cookie and get answer from GPT-4 model
    try:
        result = you.Completion.create(prompt=question)

        return result.text

    except Exception as e:
        # Return error message if an exception occurs
        return (
            f'An error occurred: {e}. Please make sure you are using a valid cloudflare clearance token and user agent.'
        )


def load_lottieurl(url: str):   
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

lotti_conding = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_p8bfn5to.json")

# Set page configuration and add header
st.set_page_config(
    page_title="gpt4freeGUI",
    initial_sidebar_state="expanded",
    page_icon="🧠",
    layout='wide',
    menu_items={
        'Get Help': 'https://github.com/xtekky/gpt4free/blob/main/README.md',
        'Report a bug': "https://github.com/xtekky/gpt4free/issues",
        'About': "### gptfree GUI",
    },
)



st.header('GPT4')


# page_bg_img = """
# <style>
# [data-testid="stAppViewContainer"]{
# background: rgb(2,0,36);
# background: radial-gradient(circle, rgba(2,0,36,1) 0%, rgba(94,47,64,1) 44%, rgba(43,76,181,1) 77%, rgba(0,212,255,1) 100%);
# }
# </style>
# """
# st.markdown(page_bg_img, unsafe_allow_html=True) 

list_of_placeholders = ["Differences between an iceberg and a glacier?", "Which stories do you know?", "What is the meaning of irony?", "Can you tell me an interesting love story?", "Can you give an accurate answer for 25% of 200?", "Which is the largest ocean in the world?", "Which story about a famous artist do you know?", "Can you differentiate between a verb and a noun?", "Explain quantum physics in 50 words without using the word quantum.", "What is a black hole?"]



question_text_area = st.text_area('Ask Any Question :', placeholder='Ski (ily)')

if st.button('🧠 Think'):
    #animation question text area going up
    c1 = time.time()
    

    with streamlit_lottie.st_lottie_spinner(lotti_conding, height=100, quality='High'):
        answer = get_answer(question_text_area)
    
    escaped = answer.encode('utf-8').decode('unicode-escape')
    # Display answer
    st.caption("Answer :")
    st.markdown(escaped)
    c2 = time.time()
    
    st.write("Characters:", len(answer)) # Did not use f strings as it looks better this way
    st.write("Time taken:", round(c2-c1, 2), "seconds")
    
    
    # if st.button('Copy 📝'): # Cant get this working
    #     try:
    #         pyperclip.copy(str(escaped))
    #     except Exception as e:
    #         st.write(f'An error occurred while trying to copy: {e}. Please try again.')
    
    


# Hide Streamlit footer
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
