from __future__ import annotations

import os
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
            '/chat/<share_id>/': {
                'function': self._share_id,
                'methods': ['GET', 'POST']
            },
            '/chat/<share_id>/<conversation_id>': {
                'function': self._share_id,
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
            '/background': {
                'function': self._background,
                'methods': ['GET']
            },
        }

    def _chat(self, conversation_id):
        if conversation_id == "share":
            return render_template('index.html', conversation_id=str(uuid.uuid4()))
        return render_template('index.html', conversation_id=conversation_id)

    def _share_id(self, share_id, conversation_id: str = ""):
        share_url = os.environ.get("G4F_SHARE_URL", "")
        conversation_id = conversation_id if conversation_id else str(uuid.uuid4())
        return render_template('index.html', share_url=share_url, share_id=share_id, conversation_id=conversation_id)

    def _index(self):
        return render_template('index.html', conversation_id=str(uuid.uuid4()))

    def _settings(self):
        return render_template('index.html', conversation_id=str(uuid.uuid4()))

    def _background(self):
        return render_template('background.html')