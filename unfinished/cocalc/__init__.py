import requests

class Completion:
    def create(prompt="What is the square root of pi",
               system_prompt="ASSUME I HAVE FULL ACCESS TO COCALC. ENCLOSE MATH IN $. INCLUDE THE LANGUAGE DIRECTLY AFTER THE TRIPLE BACKTICKS IN ALL MARKDOWN CODE BLOCKS. How can I do the following using CoCalc?") -> str:

        # Initialize a session
        session = requests.Session()
        
        # Set headers for the request
        headers = {
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Origin': 'https://cocalc.com',
            'Referer': 'https://cocalc.com/api/v2/openai/chatgpt',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        }
        session.headers.update(headers)
        
        # Set the data that will be submitted
        payload = {
            "input": prompt,
            "system": system_prompt,
            "tag": "next:index"
        }

        # Submit the request
        response = session.post("https://cocalc.com/api/v2/openai/chatgpt", json=payload).json()

        # Return the results
        return response
