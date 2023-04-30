import requests
from fake_useragent import UserAgent
from pydantic import BaseModel


class CoCalcResponse(BaseModel):
    text: str
    status: bool


class Completion:
    """A class for generating text completions using CoCalc's GPT-based chatbot."""

    API_ENDPOINT = "https://cocalc.com/api/v2/openai/chatgpt"
    DEFAULT_SYSTEM_PROMPT = "ASSUME I HAVE FULL ACCESS TO COCALC. "

    @staticmethod
    def create(prompt: str, cookie_input: str) -> CoCalcResponse:
        """
        Generate a text completion for the given prompt using CoCalc's GPT-based chatbot.

        Args:
            prompt: The text prompt to complete.
            cookie_input: The cookie required to authenticate the chatbot API request.

        Returns:
            A CoCalcResponse object containing the text completion and a boolean indicating
            whether the request was successful.
        """

        # Initialize a session with custom headers
        session = Completion._initialize_session(cookie_input)

        # Set the data that will be submitted
        payload = Completion._create_payload(prompt, Completion.DEFAULT_SYSTEM_PROMPT)

        try:
            # Submit the request and return the results
            response = session.post(Completion.API_ENDPOINT, json=payload).json()
            return CoCalcResponse(text=response['output'], status=response['success'])
        except requests.exceptions.RequestException as e:
            # Handle exceptions that may occur during the request
            print(f"Error: {e}")
            return CoCalcResponse(text="", status=False)

    @classmethod
    def _initialize_session(cls, conversation_cookie: str) -> requests.Session:
        """Initialize a session with custom headers for the request."""

        session = requests.Session()
        headers = {
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.5",
            "Origin": "https://cocalc.com",
            "Referer": "https://cocalc.com/api/v2/openai/chatgpt",
            "Cookie": conversation_cookie,
            "User-Agent": UserAgent().random,
        }
        session.headers.update(headers)

        return session

    @staticmethod
    def _create_payload(prompt: str, system_prompt: str) -> dict:
        """Create the payload for the API request."""

        return {"input": prompt, "system": system_prompt, "tag": "next:index"}
