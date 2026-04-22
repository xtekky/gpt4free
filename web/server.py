from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

UPSTREAM = os.environ.get('G4F_UPSTREAM', 'http://localhost:8080')

app = Flask(__name__)
CORS(app)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    message = data.get('message') or data.get('prompt') or ''
    if not message:
        return jsonify({'error': 'missing message'}), 400
    try:
        resp = requests.post(f"{UPSTREAM}/chat", json={'message': message}, timeout=15)
        resp.raise_for_status()
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({'reply': f"Mock reply: I received '{message[:200]}' (upstream error)."}), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
