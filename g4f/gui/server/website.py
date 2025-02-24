import uuid
from flask import render_template, redirect

def redirect_home():
    return redirect('/chat')

class Website:
    def __init__(self, app) -> None:
        self.app = app
        self.routes = {
            '/chat/': {
                'function': self._index,
                'methods': ['GET', 'POST']
            },
            '/chat/<conversation_id>': {
                'function': self._chat,
                'methods': ['GET', 'POST']
            },
            '/chat/menu/': {
                'function': redirect_home,
                'methods': ['GET', 'POST']
            },
            '/chat/settings/': {
                'function': self._settings,
                'methods': ['GET', 'POST']
            },
            '/images/': {
                'function': redirect_home,
                'methods': ['GET', 'POST']
            },
        }

    def _chat(self, conversation_id):
        if conversation_id == "share":
            return render_template('index.html', chat_id=str(uuid.uuid4()))
        return render_template('index.html', chat_id=conversation_id)

    def _index(self):
        return render_template('index.html', chat_id=str(uuid.uuid4()))

    def _settings(self):
        return render_template('index.html', chat_id=str(uuid.uuid4()))