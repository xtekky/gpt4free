from g4f.client import Client
from g4f.Provider import PuterJS

client = Client(
    provider=PuterJS,
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0IjoiYXUiLCJ2IjoiMC4wLjAiLCJ1dSI6InR2ZGJwZWZSVCtTOWJKeTZLbCszVGc9PSIsImF1Ijoib0dWODVGQjJXaENUUTNqVkwwL3FoUT09IiwicyI6IkNJSVpnT1NHajN4QW50L1lYVFg1aGc9PSIsImlhdCI6MTc0OTgyMTUyNn0.mQu8q21JwYRrzCshh9ytgx24VzDQVTNHabzDJJyB2Ws",
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    max_tokens=50,
)
print(response.choices[0].message.content)
exit()



import base64
import os
from ecdsa import SigningKey, SECP256k1
import hashlib
import requests
import uuid


# -------- Create Anonymous User --------
def create_anon_user():
    while True:
        private_key_bytes = os.urandom(32)
        try:
            sk = SigningKey.from_string(private_key_bytes, curve=SECP256k1)
            break
        except Exception:
            continue  # Try again until valid

    private_key_b64 = base64.b64encode(private_key_bytes).decode('utf-8')
    public_key_bytes = sk.verifying_key.to_string()

    # Fake server call to get user ID
    return {
        "privateKey": private_key_b64,
        "anonUserId": str(uuid.uuid4())
    }


# -------- Create Challenge Signature --------
def create_challenge(user_data):
    challenge = os.urandom(32)  # Simulated challenge
    challenge_hash = hashlib.sha256(challenge).digest()

    private_key_bytes = base64.b64decode(user_data["privateKey"])
    sk = SigningKey.from_string(private_key_bytes, curve=SECP256k1)
    signature = sk.sign(challenge_hash)

    return {
        "challenge": base64.b64encode(challenge).decode('utf-8'),
        "signature": base64.b64encode(signature).decode('utf-8'),
        "anonUserId": user_data["anonUserId"]
    }


# -------- Middleware Simulation --------
def middleware_hook(request):
    request_id = base64.b64encode(os.urandom(8)).decode()
    url_path = request["url"].split("?")[0]

    try:
        # Simulate the ef(e, t) call (fallback ID)
        statsig_id = some_external_handler(url_path, request.get("method", "GET"))
    except Exception as e:
        statsig_id = base64.b64encode(f"e:{str(e)}".encode()).decode()

    # Add headers
    headers = request.setdefault("headers", {})
    headers["x-xai-request-id"] = request_id
    headers["x-statsig-id"] = statsig_id

    return request


# -------- Simulated external call --------
def some_external_handler(path, method):
    # Simulate dynamic loading of a function
    if "fail" in path:
        raise ValueError("Simulated failure")
    return f"handler-id-for-{path}"

import base64
import urllib.parse

def rc4_decrypt(base64_str, key):
    # Step 1: Base64 decode
    decoded_bytes = base64.b64decode(base64_str)
    
    # Step 2: Percent-decode (as in JS's decodeURIComponent(escape()))
    decoded_str = urllib.parse.unquote(''.join('%%%02x' % b for b in decoded_bytes))
    
    # Step 3: RC4 Decryption
    S = list(range(256))
    j = 0
    out = []

    # KSA (Key Scheduling Algorithm)
    for i in range(256):
        j = (j + S[i] + ord(key[i % len(key)])) % 256
        S[i], S[j] = S[j], S[i]

    # PRGA (Pseudo-Random Generation Algorithm)
    i = j = 0
    for char in decoded_str:
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        S[i], S[j] = S[j], S[i]
        out.append(chr(ord(char) ^ S[(S[i] + S[j]) % 256]))

    return ''.join(out)

from Crypto.Cipher import ARC4
import base64

def rc4_decrypt(cipher_text_b64: str, key: str) -> str:
    cipher_bytes = base64.b64decode(cipher_text_b64)
    cipher = ARC4.new(key.encode())
    decrypted = cipher.decrypt(cipher_bytes)
    return decrypted.decode(errors="ignore")

def rc4_encrypt(plain_text: str, key: str) -> str:
    cipher = ARC4.new(key.encode())
    encrypted = cipher.encrypt(plain_text.encode())
    return base64.b64encode(encrypted).decode()



print(rc4_decrypt("R0QM/BrwHl93JoU09oYYeJ63W3Lzgm2wzxLaXq3F61juBwCJ36jTeZpXYbMQ87tZOra9nETi7rVfJrF/BqeiBp4y/egBRA==", "https://grok.com/rest/app-chat/conversations/new/GET"))


# -------- Usage --------
if __name__ == "__main__":
    # 1. Create user
    user = create_anon_user()
    print("Anon user created:", user)

    # 2. Create challenge signature
    challenge_data = create_challenge(user)
    print("Challenge signed:", challenge_data)

    # 3. Simulate middleware request
    request = {
        "url": "https://api.example.com/resource",
        "method": "POST",
        "headers": {}
    }
    modified_request = middleware_hook(request)
    print("Modified request headers:", modified_request["headers"])