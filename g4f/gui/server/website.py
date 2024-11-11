import uuid
from flask import render_template, redirect

class Website:
    def __init__(self, app) -> None:
        self.app = app
        def redirect_home():
            return redirect('/chat')
        self.routes = {
            '/': {
                'function': redirect_home,
                'methods': ['GET', 'POST']
            },
            '/chat/': {
                'function': self._index,
                'methods': ['GET', 'POST']
            },
            '/chat/<conversation_id>': {
                'function': self._chat,
                'methods': ['GET', 'POST']
            },
            '/menu/': {
                'function': redirect_home,
                'methods': ['GET', 'POST']
            },
            '/settings/': {
                'function': redirect_home,
                'methods': ['GET', 'POST']
            },
            '/images/': {
                'function': redirect_home,
                'methods': ['GET', 'POST']
            },
        }

    def _chat(self, conversation_id):
        if '-' not in conversation_id:
            return redirect('/chat')
        return render_template('index.html', chat_id=conversation_id)

    def _index(self):
        return render_template('index.html', chat_id=str(uuid.uuid4()))
