import streamlit as st
import phind


class GPT4Model:
    def __init__(self, clearance, user_agent):
        self.clearance = clearance
        self.user_agent = user_agent
        phind.cloudflare_clearance = self.clearance
        phind.phind_api = self.user_agent

    def get_answer(self, question: str) -> str:
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
            return f'An error occurred: {e}. Please make sure you are using a valid cloudflare clearance token and user agent.'


def create_streamlit_interface(gpt4_model: GPT4Model):
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

    question_text_area = st.text_area(
        '🤖 Ask Any Question :', placeholder='Explain quantum computing in 50 words')
    if st.button('🧠 Think'):
        answer = gpt4_model.get_answer(question_text_area)
        st.caption("Answer :")
        st.markdown(answer)

    hide_streamlit_style = """
                <style>
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


if __name__ == "__main__":
    clearance = ""
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
    gpt4_model = GPT4Model(clearance, user_agent)
    create_streamlit_interface(gpt4_model)
