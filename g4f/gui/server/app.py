from flask import Flask

def create_app() -> Flask:
    app = Flask(__name__)
    return app