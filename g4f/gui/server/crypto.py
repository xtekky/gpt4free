from __future__ import annotations

import os
import base64

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey, RSAPrivateKey

from ...cookies import get_cookies_dir

def create_or_read_keys() -> tuple[RSAPrivateKey, RSAPublicKey]:
    private_key_file = os.path.join(get_cookies_dir(), "private_key.pem")
    public_key_file = os.path.join(get_cookies_dir(), "public_key.pem")

    if os.path.isfile(private_key_file) and os.path.isfile(public_key_file):
        # Read private key
        with open(private_key_file, 'rb') as f:
            private_key_pem = f.read()
            private_key = serialization.load_pem_private_key(
                private_key_pem,
                password=None  # Use password=b'mypassword' here if the key is encrypted
            )

        # Read public key
        with open(public_key_file, 'rb') as f:
            public_key_pem = f.read()
            public_key = serialization.load_pem_public_key(public_key_pem)

        return private_key, public_key

    # Generate keys
    private_key_obj = rsa.generate_private_key(public_exponent=65537, key_size=1024)
    public_key_obj = private_key_obj.public_key()

    # Serialize private key
    private_key_pem = private_key_obj.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    # Serialize public key
    public_key_pem = public_key_obj.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    # Write the private key PEM to a file
    with open(private_key_file, 'wb') as f:
        f.write(private_key_pem)

    # Write the public key PEM to a file
    with open(public_key_file, 'wb') as f:
        f.write(public_key_pem)

    return private_key_obj, public_key_obj

def decrypt_data(private_key_obj: RSAPrivateKey, encrypted_data: str) -> str:
    decrypted = private_key_obj.decrypt(
        base64.b64decode(encrypted_data),
        padding.PKCS1v15()
    )
    return decrypted.decode()

def encrypt_data(public_key: RSAPublicKey, decrypted_data: str) -> str:
    encrypted = public_key.encrypt(
        decrypted_data.encode(),
        padding.PKCS1v15()
    )
    return base64.b64encode(encrypted).decode()