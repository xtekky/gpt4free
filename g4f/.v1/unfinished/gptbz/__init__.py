from json import dumps, loads

import websockets


# Define the asynchronous function to test the WebSocket connection


async def test():
    # Establish a WebSocket connection with the specified URL
    async with websockets.connect('wss://chatgpt.func.icu/conversation+ws') as wss:

        # Prepare the message payload as a JSON object
        payload = {
            'content_type': 'text',
            'engine': 'chat-gpt',
            'parts': ['hello world'],
            'options': {}
        }

        # Send the payload to the WebSocket server
        await wss.send(dumps(obj=payload, separators=(',', ':')))

        # Initialize a variable to track the end of the conversation
        ended = None

        # Continuously receive and process messages until the conversation ends
        while not ended:
            try:
                # Receive and parse the JSON response from the server
                response = await wss.recv()
                json_response = loads(response)

                # Print the entire JSON response
                print(json_response)

                # Check for the end of the conversation
                ended = json_response.get('eof')

                # If the conversation has not ended, print the received message
                if not ended:
                    print(json_response['content']['parts'][0])

            # Handle cases when the connection is closed by the server
            except websockets.ConnectionClosed:
                break
