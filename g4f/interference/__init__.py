import json
import time
import random
import string

from typing       import Any
from flask        import Flask, request
from flask_cors   import CORS
from g4f          import ChatCompletion

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return 'interference api, url: http://127.0.0.1:1337'

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

def run_interference():
    app.run(host='0.0.0.0', port=1337, debug=True)