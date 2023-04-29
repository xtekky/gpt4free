import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth


def auth():
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    name, authentication_status, username = authenticator.login('Login', 'main')

    return name, authentication_status, username, authenticator

def gen_pwd_hash(pwd):
    hashed_passwords = stauth.Hasher([pwd]).generate()

    print(hashed_passwords[0])


if __name__ == "__main__":
    gen_pwd_hash("gpt4free")