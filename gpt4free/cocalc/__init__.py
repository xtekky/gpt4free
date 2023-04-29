import requests
from fake_useragent import UserAgent
from pydantic import BaseModel


class CoCalcResponse(BaseModel):
    text: str
    status: bool


class Completion:
    @staticmethod
    def create(prompt: str, cookie_input: str) -> CoCalcResponse:
        # Initialize a session with custom headers
        session = Completion._initialize_session(cookie_input)

        # Set the data that will be submitted
        payload = Completion._create_payload(prompt, 'ASSUME I HAVE FULL ACCESS TO COCALC. ')

        # Submit the request and return the results
        return Completion._submit_request(session, payload)

    @classmethod
    def _initialize_session(cls, conversation_cookie) -> requests.Session:
        """Initialize a session with custom headers for the request."""

        session = requests.Session()
        headers = {
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Origin': 'https://cocalc.com',
            'Referer': 'https://cocalc.com/api/v2/openai/chatgpt',
            'Cookie': conversation_cookie,
            'User-Agent': UserAgent().random,
        }
        session.headers.update(headers)

        return session

    @staticmethod
    def _create_payload(prompt: str, system_prompt: str) -> dict:
        return {'input': prompt, 'system': system_prompt, 'tag': 'next:index'}

    @staticmethod
    def _submit_request(session: requests.Session, payload: dict) -> CoCalcResponse:
        response = session.post('https://cocalc.com/api/v2/openai/chatgpt', json=payload).json()
        return CoCalcResponse(text=response['output'], status=response['success'])
