import jwt
from   datetime import datetime, timedelta
# from   cryptography.hazmat.primitives import serialization
# from   cryptography.hazmat.primitives.serialization import load_pem_private_key
# from   cryptography.hazmat.backends import default_backend


def do_jwt(json_data: dict):

    private_key = b'''-----BEGIN RSA PRIVATE KEY-----
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
-----END RSA PRIVATE KEY-----'''

    header = {
        'alg': 'RS256',
        'typ': 'JWT',
        'kid': '0d1bb0d7-45e4-445c-889e-57419470a570'
    }

    payload = {
        **json_data,
        'iat': int(datetime.now().timestamp()),
        'exp': int((datetime.now() + timedelta(minutes=10)).timestamp()),
        'iss': 'https://ora.sh'
    }

    return jwt.encode(payload, private_key, algorithm='RS256', headers=header)