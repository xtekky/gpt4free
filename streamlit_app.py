import streamlit as st
import phind

phind.cf_clearance = ''
phind.user_agent   = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'

def phind_get_answer(question:str)->str:
    # set cf_clearance cookie
    try:
    
        result = phind.Completion.create(
        model  = 'gpt-4',
        prompt = question,
        results     = phind.Search.create(question, actualSearch = True),
        creative    = False,
        detailed    = False,
        codeContext = '') 
        return result.completion.choices[0].text

    except Exception as e:
        return 'An error occured, please make sure you are using a cf_clearance token and correct useragent | %s' % e

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