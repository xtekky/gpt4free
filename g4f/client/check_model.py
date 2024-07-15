import g4f
from g4f.client import Client
import time
import asyncio
import aiohttp

class CheckModel:
    def __init__(self, model):
        """
        Initializes the CheckModel class.

        Args:
            model (str): The model to be checked.

        Returns:
            None
        """
        self.model = model
        self.client = Client()
        self.working_providers = [
            provider.__name__
            for provider in g4f.Provider.__providers__
            if provider.working
        ]

    async def get_response(self, provider):
        """
        Asynchronously gets a response from a chat provider.

        Args:
            provider (str): The name of the chat provider.

        Returns:
            None

        Raises:
            Exception: If an error occurs during the chat completion.

        This function sends a chat message to the specified provider and retrieves the response. The response is saved to a text file.
        If an error occurs during the chat completion, the error is also saved to the text file.

        Example usage:
        ```
        await get_response('OpenAI')
        ```
        """
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "Hi, How are you"}],
                    provider=provider,
                    session=session,
                )
            end_time = time.time()
            elapsed_time = end_time - start_time
            response_content = response.choices[0].message.content
            result = f"Model: {self.model}\nProvider: {provider}\nResponse: {response_content}\nTime taken: {elapsed_time:.2f} seconds\n"
        except Exception as e:
            elapsed_time = time.time() - start_time
            result = f"Model: {self.model}\nProvider: {provider}\nError: {str(e)}\nTime taken: {elapsed_time:.2f} seconds\n"

        # Save each result to the text file
        with open('model_responses.txt', 'a') as file:
            file.write(result + "\n")


    async def check_model(self):
        """
        Asynchronously checks the model by gathering responses from different providers.
        """
        print(f'''
                ==========================
                Start for {self.model}....
                ==========================
                ''')
        tasks = [self.get_response(provider) for provider in self.working_providers]
        await asyncio.gather(*tasks)

