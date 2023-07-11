import os,sys
import json
import subprocess
# from ...typing import sha256, Dict, get_type_hints

url = 'https://b.ai-huan.xyz'
model = ['gpt-3.5-turbo', 'gpt-4']
supports_stream = True
needs_auth = False

def _create_completion(model: str, messages: list, stream: bool, **kwargs):
    path = os.path.dirname(os.path.realpath(__file__))
    config = json.dumps({
        'messages': messages,
        'model': model}, separators=(',', ':'))
    cmd = ['python', f'{path}/helpers/binghuan.py', config]

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    for line in iter(p.stdout.readline, b''):
        yield line.decode('cp1252')
    


# params = f'g4f.Providers.{os.path.basename(__file__)[:-3]} supports: ' + \
#     '(%s)' % ', '.join(
#         [f"{name}: {get_type_hints(_create_completion)[name].__name__}" for name in _create_completion.__code__.co_varnames[:_create_completion.__code__.co_argcount]])


# Temporary For ChatCompletion Class
class ChatCompletion:
    @staticmethod
    def create(model: str, messages: list, provider: None or str, stream: bool = False, auth: str = False, **kwargs):
        kwargs['auth'] = auth

        if provider and needs_auth and not auth:
            print(
                f'ValueError: {provider} requires authentication (use auth="cookie or token or jwt ..." param)', file=sys.stderr)
            sys.exit(1)

        try:
            return (_create_completion(model, messages, stream, **kwargs)
                    if stream else ''.join(_create_completion(model, messages, stream, **kwargs)))
        except TypeError as e:
            print(e)
            arg: str = str(e).split("'")[1]
            print(
                f"ValueError: {provider} does not support '{arg}' argument", file=sys.stderr)
            sys.exit(1)