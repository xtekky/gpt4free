import requests


class Completion:
    def create(self, prompt="What is the square root of pi",
               system_prompt=("ASSUME I HAVE FULL ACCESS TO COCALC. ENCLOSE MATH IN $. "
                              "INCLUDE THE LANGUAGE DIRECTLY AFTER THE TRIPLE BACKTICKS "
                              "IN ALL MARKDOWN CODE BLOCKS. How can I do the following using CoCalc?")) -> str:
        # Initialize a session with custom headers
        session = self._initialize_session()

        # Set the data that will be submitted
        payload = self._create_payload(prompt, system_prompt)

        # Submit the request and return the results
        return self._submit_request(session, payload)

    def _initialize_session(self) -> requests.Session:
        """Initialize a session with custom headers for the request."""

        session = requests.Session()
        headers = {
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Origin': 'https://cocalc.com',
            'Referer': 'https://cocalc.com/api/v2/openai/chatgpt',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        }
        session.headers.update(headers)

        return session

    def _create_payload(self, prompt: str, system_prompt: str) -> dict:
        """Create the payload with the given prompts."""

        return {
            "input": prompt,
            "system": system_prompt,
            "tag": "next:index"
        }

    def _submit_request(self, session: requests.Session, payload: dict) -> str:
        """Submit the request to the API and return the response."""

        response = session.post(
            "https://cocalc.com/api/v2/openai/chatgpt", json=payload).json()
        return response
