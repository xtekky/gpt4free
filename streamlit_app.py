import streamlit as st
import phind

def phind_get_answer(question:str)->str:
    # set cf_clearance cookie
    phind.cf_clearance = 'heguhSRBB9d0sjLvGbQECS8b80m2BQ31xEmk9ChshKI-1682268995-0-160'
    phind.user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'
    result = phind.Completion.create(
    model  = 'gpt-4',
    prompt = question,
    results     = phind.Search.create(question, actualSearch = True),
    creative    = False,
    detailed    = False,
    codeContext = '') 
    return result.completion.choices[0].text


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

question_text_area = st.text_area('🤖 Ask Any Question :', placeholder='Explain quantum computing in 50 words')
if st.button('🧠 Think'):
    answer = phind_get_answer(question_text_area)
    st.caption("Answer :")
    st.markdown(answer)


hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 