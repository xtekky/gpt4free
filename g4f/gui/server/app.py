import sys, os
from flask import Flask

if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, "client/html")
else:
    template_folder = "./../client/html"

app = Flask(__name__, template_folder=template_folder)