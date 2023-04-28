import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import streamlit as st
from streamlit_chat import message
from query_methods import query, avail_query_methods
import pickle
import openai_rev

conversations_file = "conversations.pkl"

def load_conversations():
    try:
        with open(conversations_file, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []

def save_conversations(conversations, current_conversation):
    updated = False
    for i, conversation in enumerate(conversations):
        if conversation == current_conversation:
            conversations[i] = current_conversation
            updated = True
            break
    if not updated:
        conversations.append(current_conversation)
    with open(conversations_file, "wb") as f:
        pickle.dump(conversations, f)

st.header("Chat Placeholder")

if 'conversations' not in st.session_state:
    st.session_state['conversations'] = load_conversations()

if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ''

if 'selected_conversation' not in st.session_state:
    st.session_state['selected_conversation'] = None

if 'input_field_key' not in st.session_state:
    st.session_state['input_field_key'] = 0
    
if 'query_method' not in st.session_state:
    st.session_state['query_method'] = query

# Initialize new conversation
if 'current_conversation' not in st.session_state or st.session_state['current_conversation'] is None:
    st.session_state['current_conversation'] = {'user_inputs': [], 'generated_responses': []}


input_placeholder = st.empty()
user_input = input_placeholder.text_input('You:', key=f'input_text_{len(st.session_state["current_conversation"]["user_inputs"])}')
submit_button = st.button("Submit")

if user_input or submit_button:
    output = query(user_input, st.session_state['query_method'])
    
    st.session_state.current_conversation['user_inputs'].append(user_input)
    st.session_state.current_conversation['generated_responses'].append(output)
    save_conversations(st.session_state.conversations, st.session_state.current_conversation)
    user_input = input_placeholder.text_input('You:', value='', key=f'input_text_{len(st.session_state["current_conversation"]["user_inputs"])}')  # Clear the input field


# Add a button to create a new conversation
if st.sidebar.button("New Conversation"):
    st.session_state['selected_conversation'] = None
    st.session_state['current_conversation'] = {'user_inputs': [], 'generated_responses': []}
    st.session_state['input_field_key'] += 1

st.session_state['query_method'] = st.sidebar.selectbox(
    "Select API:",
    options=openai_rev.Provider.__members__.keys(),
    index=0
)

# Sidebar
st.sidebar.header("Conversation History")

for i, conversation in enumerate(st.session_state.conversations):
    if st.sidebar.button(f"Conversation {i + 1}: {conversation['user_inputs'][0]}", key=f"sidebar_btn_{i}"):
        st.session_state['selected_conversation'] = i
        st.session_state['current_conversation'] = st.session_state.conversations[i]

if st.session_state['selected_conversation'] is not None:
    conversation_to_display = st.session_state.conversations[st.session_state['selected_conversation']]
else:
    conversation_to_display = st.session_state.current_conversation

if conversation_to_display['generated_responses']:
    for i in range(len(conversation_to_display['generated_responses']) - 1, -1, -1):
        message(conversation_to_display["generated_responses"][i], key=f"display_generated_{i}")
        message(conversation_to_display['user_inputs'][i], is_user=True, key=f"display_user_{i}")
