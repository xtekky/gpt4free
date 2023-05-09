from flask import Flask, request, jsonify


import os
import sys

app = Flask(__name__)
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

# import streamlit as st
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

@app.route('/')
def index():
    return 'Ready'

@app.route('/api/chatgpt', methods=['POST'])
def api():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    question = data['question']
    answer = get_answer(question)
    escaped = answer.encode('utf-8').decode('unicode-escape')
    # return jsonify({'answer': escaped})
    return {'answer': escaped}


if __name__ == '__main__':
    app.run()

    