import atexit
import Levenshtein
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import streamlit as st
from streamlit_chat import message
from query_methods import query, avail_query_methods
import pickle

conversations_file = "conversations.pkl"

def load_conversations():
    try:
        with open(conversations_file, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []
    except EOFError:
        return []


def save_conversations(conversations, current_conversation):
    updated = False
    for idx, conversation in enumerate(conversations):
        if conversation == current_conversation:
            conversations[idx] = current_conversation
            updated = True
            break
    if not updated:
        conversations.append(current_conversation)

    temp_conversations_file = "temp_" + conversations_file
    with open(temp_conversations_file, "wb") as f:
        pickle.dump(conversations, f)

    os.replace(temp_conversations_file, conversations_file)

def delete_conversation(conversations, current_conversation):
    for idx, conversation in enumerate(conversations):
        conversations[idx] = current_conversation
        break
    conversations.remove(current_conversation)

    temp_conversations_file = "temp_" + conversations_file
    with open(temp_conversations_file, "wb") as f:
        pickle.dump(conversations, f)

    os.replace(temp_conversations_file, conversations_file)

def exit_handler():
    print("Exiting, saving data...")
    # Perform cleanup operations here, like saving data or closing open files.
    save_conversations(st.session_state.conversations, st.session_state.current_conversation)


# Register the exit_handler function to be called when the program is closing.
atexit.register(exit_handler)

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

if 'search_query' not in st.session_state:
    st.session_state['search_query'] = ''

# Initialize new conversation
if 'current_conversation' not in st.session_state or st.session_state['current_conversation'] is None:
    st.session_state['current_conversation'] = {'user_inputs': [], 'generated_responses': []}

input_placeholder = st.empty()
user_input = input_placeholder.text_input(
    'You:', value=st.session_state['input_text'], key=f'input_text_-1'#{st.session_state["input_field_key"]}
)
submit_button = st.button("Submit")

if (user_input and user_input != st.session_state['input_text']) or submit_button:
    output = query(user_input, st.session_state['query_method'])

    escaped_output = output.encode('utf-8').decode('unicode-escape')

    st.session_state['current_conversation']['user_inputs'].append(user_input)
    st.session_state.current_conversation['generated_responses'].append(escaped_output)
    save_conversations(st.session_state.conversations, st.session_state.current_conversation)
    st.session_state['input_text'] = ''
    st.session_state['input_field_key'] += 1  # Increment key value for new widget
    user_input = input_placeholder.text_input(
        'You:', value=st.session_state['input_text'], key=f'input_text_{st.session_state["input_field_key"]}'
    )  # Clear the input field

# Add a button to create a new conversation
if st.sidebar.button("New Conversation"):
    st.session_state['selected_conversation'] = None
    st.session_state['current_conversation'] = {'user_inputs': [], 'generated_responses': []}
    st.session_state['input_field_key'] += 1  # Increment key value for new widget
    st.session_state['query_method'] = st.sidebar.selectbox("Select API:", options=avail_query_methods, index=0)

# Proxy
st.session_state['proxy'] = st.sidebar.text_input("Proxy: ")

# Searchbar
search_query = st.sidebar.text_input("Search Conversations:", value=st.session_state.get('search_query', ''), key='search')

if search_query:
    filtered_conversations = []
    indices = []
    for idx, conversation in enumerate(st.session_state.conversations):
        if search_query in conversation['user_inputs'][0]:
            filtered_conversations.append(conversation)
            indices.append(idx)

    filtered_conversations = list(zip(indices, filtered_conversations))
    conversations = sorted(filtered_conversations, key=lambda x: Levenshtein.distance(search_query, x[1]['user_inputs'][0]))

    sidebar_header = f"Search Results ({len(conversations)})"
else:
    conversations = st.session_state.conversations
    sidebar_header = "Conversation History"

# Sidebar
st.sidebar.header(sidebar_header)
sidebar_col1, sidebar_col2 = st.sidebar.columns([5,1])
for idx, conversation in enumerate(conversations):
    if sidebar_col1.button(f"Conversation {idx + 1}: {conversation['user_inputs'][0]}", key=f"sidebar_btn_{idx}"):
        st.session_state['selected_conversation'] = idx
        st.session_state['current_conversation'] = conversation
    if sidebar_col2.button('🗑️', key=f"sidebar_btn_delete_{idx}"):
        if st.session_state['selected_conversation'] == idx:
            st.session_state['selected_conversation'] = None
            st.session_state['current_conversation'] = {'user_inputs': [], 'generated_responses': []}
        delete_conversation(conversations, conversation)
        st.experimental_rerun()
if st.session_state['selected_conversation'] is not None:
    conversation_to_display = conversations[st.session_state['selected_conversation']]
else:
    conversation_to_display = st.session_state.current_conversation

if conversation_to_display['generated_responses']:
    for i in range(len(conversation_to_display['generated_responses']) - 1, -1, -1):
        message(conversation_to_display["generated_responses"][i], key=f"display_generated_{i}")
        message(conversation_to_display['user_inputs'][i], is_user=True, key=f"display_user_{i}")