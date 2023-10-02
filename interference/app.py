import json
import time
import random
import string
import requests

from typing       import Any
from flask        import Flask, request
from flask_cors   import CORS
from transformers import AutoTokenizer
from g4f          import ChatCompletion

app = Flask(__name__)
CORS(app)

@app.route('/chat/completions', methods=['POST'])
def chat_completions():
    model    = request.get_json().get('model', 'gpt-3.5-turbo')
    stream   = request.get_json().get('stream', False)
    messages = request.get_json().get('messages')

    response = ChatCompletion.create(model = model, 
                                     stream = stream, messages = messages)

    completion_id = ''.join(random.choices(string.ascii_letters + string.digits, k=28))
    completion_timestamp = int(time.time())

    if not stream:
        return {
            'id': f'chatcmpl-{completion_id}',
            'object': 'chat.completion',
            'created': completion_timestamp,
            'model': model,
            'choices': [
                {
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': response,
                    },
                    'finish_reason': 'stop',
                }
            ],
            'usage': {
                'prompt_tokens': None,
                'completion_tokens': None,
                'total_tokens': None,
            },
        }

    def streaming():
        for chunk in response:
            completion_data = {
                'id': f'chatcmpl-{completion_id}',
                'object': 'chat.completion.chunk',
                'created': completion_timestamp,
                'model': model,
                'choices': [
                    {
                        'index': 0,
                        'delta': {
                            'content': chunk,
                        },
                        'finish_reason': None,
                    }
                ],
            }

            content = json.dumps(completion_data, separators=(',', ':'))
            yield f'data: {content}\n\n'
            time.sleep(0.1)

        end_completion_data: dict[str, Any] = {
            'id': f'chatcmpl-{completion_id}',
            'object': 'chat.completion.chunk',
            'created': completion_timestamp,
            'model': model,
            'choices': [
                {
                    'index': 0,
                    'delta': {},
                    'finish_reason': 'stop',
                }
            ],
        }
        content = json.dumps(end_completion_data, separators=(',', ':'))
        yield f'data: {content}\n\n'

    return app.response_class(streaming(), mimetype='text/event-stream')


# Get the embedding from huggingface
def get_embedding(input_text, token):
    huggingface_token = token
    embedding_model = 'sentence-transformers/all-mpnet-base-v2'
    max_token_length = 500

    # Load the tokenizer for the 'all-mpnet-base-v2' model
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    # Tokenize the text and split the tokens into chunks of 500 tokens each
    tokens = tokenizer.tokenize(input_text)
    token_chunks = [tokens[i:i + max_token_length]
                    for i in range(0, len(tokens), max_token_length)]

    # Initialize an empty list
    embeddings = []

    # Create embeddings for each chunk
    for chunk in token_chunks:
        # Convert the chunk tokens back to text
        chunk_text = tokenizer.convert_tokens_to_string(chunk)

        # Use the Hugging Face API to get embeddings for the chunk
        api_url = f'https://api-inference.huggingface.co/pipeline/feature-extraction/{embedding_model}'
        headers = {'Authorization': f'Bearer {huggingface_token}'}
        chunk_text = chunk_text.replace('\n', ' ')

        # Make a POST request to get the chunk's embedding
        response = requests.post(api_url, headers=headers, json={
                                 'inputs': chunk_text, 'options': {'wait_for_model': True}})

        # Parse the response and extract the embedding
        chunk_embedding = response.json()
        # Append the embedding to the list
        embeddings.append(chunk_embedding)

    # averaging all the embeddings
    # this isn't very effective
    # someone a better idea?
    num_embeddings = len(embeddings)
    average_embedding = [sum(x) / num_embeddings for x in zip(*embeddings)]
    embedding = average_embedding
    return embedding


@app.route('/embeddings', methods=['POST'])
def embeddings():
    input_text_list = request.get_json().get('input')
    input_text      = ' '.join(map(str, input_text_list))
    token           = request.headers.get('Authorization').replace('Bearer ', '')
    embedding       = get_embedding(input_text, token)
    
    return {
        'data': [
            {
                'embedding': embedding,
                'index': 0,
                'object': 'embedding'
            }
        ],
        'model': 'text-embedding-ada-002',
        'object': 'list',
        'usage': {
            'prompt_tokens': None,
            'total_tokens': None
        }
    }

def main():
    app.run(host='0.0.0.0', port=1337, debug=True)

if __name__ == '__main__':
    main()