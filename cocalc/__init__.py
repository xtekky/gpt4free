import requests

class Completion:
    @staticmethod
    def create(prompt:str, cookieInput:str) -> str:
        # Initialize a session with custom headers
        session = Completion._initialize_session(cookieInput)

        # Set the data that will be submitted
        payload = Completion._create_payload(prompt, ("ASSUME I HAVE FULL ACCESS TO COCALC. "))

        # Submit the request and return the results
        return Completion._submit_request(session, payload)

    @classmethod
    def _initialize_session(cls, conversationCookie) -> requests.Session:
        """Initialize a session with custom headers for the request."""

        session = requests.Session()
        headers = {
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Origin': 'https://cocalc.com',
            'Referer': 'https://cocalc.com/api/v2/openai/chatgpt',
            'Cookie': conversationCookie,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        }
        session.headers.update(headers)

        return session

    @classmethod
    def _create_payload(
        cls,
        prompt: str, 
        system_prompt: str
        ) -> dict:

        return {
            "input": prompt,
            "system": system_prompt,
            "tag": "next:index"
        }

    @classmethod
    def _submit_request(
        cls,
        session: requests.Session, 
        payload: dict
        ) -> str:

        response = session.post(
            "https://cocalc.com/api/v2/openai/chatgpt", json=payload).json()
        return {
            "response":response["output"],
            "success":response["success"]
        }