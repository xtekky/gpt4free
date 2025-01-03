import sys, os
from flask import Flask

def create_app() -> Flask:
    if getattr(sys, 'frozen', False):
        template_folder = os.path.join(sys._MEIPASS, "client")
    else:
        template_folder = "../client"
    app = Flask(__name__, template_folder=template_folder, static_folder=f"{template_folder}/static")
    app.config["TEMPLATES_AUTO_RELOAD"] = True  # Enable auto reload in debug mode
    return app