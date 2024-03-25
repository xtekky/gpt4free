import json
import base64
import hashlib
import random
from Crypto.Cipher import AES

def pad(data: str) -> bytes:
    # Convert the string to bytes and calculate the number of bytes to pad
    data_bytes = data.encode()
    padding = 16 - (len(data_bytes) % 16)
    # Append the padding bytes with their value
    return data_bytes + bytes([padding] * padding)

def encrypt(data, key):
    salt = ""
    salted = ""
    dx = bytes()

    # Generate salt, as 8 random lowercase letters
    salt = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(8))

    # Our final key and IV come from the key and salt being repeatedly hashed
    for x in range(3):
        dx = hashlib.md5(dx + key.encode() + salt.encode()).digest()
        salted += dx.hex()

    # Pad the data before encryption
    data = pad(data)

    aes = AES.new(
        bytes.fromhex(salted[:64]), AES.MODE_CBC, bytes.fromhex(salted[64:96])
    )

    return json.dumps(
        {
            "ct": base64.b64encode(aes.encrypt(data)).decode(),
            "iv": salted[64:96],
            "s": salt.encode().hex(),
        }
    )

def unpad(data: bytes) -> bytes:
    # Extract the padding value from the last byte and remove padding
    padding_value = data[-1]
    return data[:-padding_value]

def decrypt(data: str, key: str):
    # Parse JSON data
    parsed_data = json.loads(base64.b64decode(data))
    ct = base64.b64decode(parsed_data["ct"])
    iv = bytes.fromhex(parsed_data["iv"])
    salt = bytes.fromhex(parsed_data["s"])

    salted = ''
    dx = b''
    for x in range(3):
        dx = hashlib.md5(dx + key.encode() + salt).digest()
        salted += dx.hex()
        
    aes = AES.new(
        bytes.fromhex(salted[:64]), AES.MODE_CBC, iv
    )

    data = aes.decrypt(ct)
    if data.startswith(b'[{"key":'):
        return unpad(data).decode()