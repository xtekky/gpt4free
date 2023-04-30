import atexit
import os
import sys
import pickle
import streamlit as st
from streamlit_chat import message
from query_methods import query, avail_query_methods

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))


class ConversationManager:
    def __init__(self, conversations_file="conversations.pkl"):
        self.conversations_file = conversations_file
        self.conversations = self.load_conversations()

    def load_conversations(self):
        try:
            with open(self.conversations_file, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return []

    def save_conversations(self):
        temp_conversations_file = "temp_" + self.conversations_file
        with open(temp_conversations_file, "wb") as f:
            pickle.dump(self.conversations, f)
        os.replace(temp_conversations_file, self.conversations_file)

    def update_conversation(self, current_conversation):
        updated = False
        for idx, conversation in enumerate(self.conversations):
            if conversation == current_conversation:
                self.conversations[idx] = current_conversation
                updated = True
                break
        if not updated:
            self.conversations.append(current_conversation)

    def exit_handler(self):
        print("Exiting, saving data...")
        self.save_conversations()


conversation_manager = ConversationManager()
atexit.register(conversation_manager.exit_handler)

st.header("Chat Placeholder")

st.session_state.setdefault('input_text', '')
st.session_state.setdefault('selected_conversation', None)
st.session_state.setdefault('input_field_key', 0)
st.session_state.setdefault('query_method', query)
st.session_state.setdefault('current_conversation', {
                            'user_inputs': [], 'generated_responses': []})

input_placeholder = st.empty()
user_input = input_placeholder.text_input(
    'You:', value=st.session_state['input_text'], key=f'input_text_{st.session_state["input_field_key"]}'
)
submit_button = st.button("Submit")

if (user_input and user_input != st.session_state['input_text']) or submit_button:
    output = query(user_input, st.session_state['query_method'])

    escaped_output = output.encode('utf-8').decode('unicode-escape')

    st.session_state.current_conversation['user_inputs'].append(user_input)
    st.session_state.current_conversation['generated_responses'].append(
        escaped_output)
    conversation_manager.update_conversation(
        st.session_state.current_conversation)
    st.session_state['input_text'] = ''
    user_input = input_placeholder.text_input(
        'You:', value=st.session_state['input_text'], key=f'input_text_{st.session_state["input_field_key"]}'
    )

if st.sidebar.button("New Conversation"):
    st.session_state['selected_conversation'] = None
    st.session_state['current_conversation'] = {
        'user_inputs': [], 'generated_responses': []}
    st.session_state['input_field_key'] += 1

st.session_state['query_method'] = st.sidebar.selectbox(
    "Select API:", options=avail_query_methods, index=0)
st.session_state['proxy'] = st.sidebar.text_input("Proxy: ")

st.sidebar.header("Conversation History")

for idx, conversation in enumerate(conversation_manager.conversations):
    if st.sidebar.button(f"Conversation {idx + 1}: {conversation['user_inputs'][0]}", key=f"sidebar_btn_{idx}"):
        st.session_state['selected_conversation'] = idx
        st.session_state['current_conversation'] = conversation_manager.conversations[idx]

conversation_to_display = (
    st.session_state.current_conversation
    if st.session_state['selected_conversation'] is None
    else conversation_manager.conversations[st.session_state['selected_conversation']]
)

if conversation_to_display['generated_responses']:
    for i in range(len(conversation_to_display['generated_responses']) - 1, -1, -1):
        message(
            conversation_to_display["generated_responses"][i], key=f"display_generated_{i}")
        message(conversation_to_display['user_inputs']
                [i], is_user=True, key=f"display_user_{i}")
