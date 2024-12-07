# G4F Authentication Setup Guide

This documentation explains how to set up Basic Authentication for the GUI and API key authentication for the API when running the G4F server.

## Prerequisites

Before proceeding, ensure you have the following installed:
- Python 3.x
- G4F package installed (ensure it is set up and working)
- Basic knowledge of using environment variables on your operating system

## Steps to Set Up Authentication

### 1. API Key Authentication for Both GUI and API

To secure both the GUI and the API, you'll authenticate using an API key. The API key should be injected via an environment variable and passed to both the GUI (via Basic Authentication) and the API.

#### Steps to Inject the API Key Using Environment Variables:

1. **Set the environment variable** for your API key:

   On Linux/macOS:
   ```bash
   export G4F_API_KEY="your-api-key-here"
   ```

   On Windows (Command Prompt):
   ```bash
   set G4F_API_KEY="your-api-key-here"
   ```

   On Windows (PowerShell):
   ```bash
   $env:G4F_API_KEY="your-api-key-here"
   ```

   Replace `your-api-key-here` with your actual API key.

2. **Run the G4F server with the API key injected**:

   Use the following command to start the G4F server. The API key will be passed to both the GUI and the API:

   ```bash
   python -m g4f --debug --port 8080 --g4f-api-key $G4F_API_KEY
   ```

   - `--debug` enables debug mode for more verbose logs.
   - `--port 8080` specifies the port on which the server will run (you can change this if needed).
   - `--g4f-api-key` specifies the API key for both the GUI and the API.

#### Example:

```bash
export G4F_API_KEY="my-secret-api-key"
python -m g4f --debug --port 8080 --g4f-api-key $G4F_API_KEY
```

Now, both the GUI and API will require the correct API key for access.

---

### 2. Accessing the GUI with Basic Authentication

The GUI uses **Basic Authentication**, where the **username** can be any value, and the **password** is your API key.

#### Example:

To access the GUI, open your web browser and navigate to `http://localhost:8080/chat/`. You will be prompted for a username and password.

- **Username**: You can use any username (e.g., `user` or `admin`).
- **Password**: Enter your API key (the same key you set in the `G4F_API_KEY` environment variable).

---

### 3. Python Example for Accessing the API

To interact with the API, you can send requests by including the `g4f-api-key` in the headers. Here's an example of how to do this using the `requests` library in Python.

#### Example Code to Send a Request:

```python
import requests

url = "http://localhost:8080/v1/chat/completions"

# Body of the request
body = {
    "model": "your-model-name",  # Replace with your model name
    "provider": "your-provider",  # Replace with the provider name
    "messages": [
        {
            "role": "user",
            "content": "Hello"
        }
    ]
}

# API Key (can be set as an environment variable)
api_key = "your-api-key-here"  # Replace with your actual API key

# Send the POST request
response = requests.post(url, json=body, headers={"g4f-api-key": api_key})

# Check the response
print(response.status_code)
print(response.json())
```

In this example:
- Replace `"your-api-key-here"` with your actual API key.
- `"model"` and `"provider"` should be replaced with the appropriate model and provider you're using.
- The `messages` array contains the conversation you want to send to the API.

#### Response:

The response will contain the output of the API request, such as the model's completion or other relevant data, which you can then process in your application.

---

### 4. Testing the Setup

- **Accessing the GUI**: Open a web browser and navigate to `http://localhost:8080/chat/`. The GUI will now prompt you for a username and password. You can enter any username (e.g., `admin`), and for the password, enter the API key you set up in the environment variable.
  
- **Accessing the API**: Use the Python code example above to send requests to the API. Ensure the correct API key is included in the `g4f-api-key` header.

---

### 5. Troubleshooting

- **GUI Access Issues**: If you're unable to access the GUI, ensure that you are using the correct API key as the password.
- **API Access Issues**: If the API is rejecting requests, verify that the `G4F_API_KEY` environment variable is correctly set and passed to the server. You can also check the server logs for more detailed error messages.

---

## Summary

By following the steps above, you will have successfully set up Basic Authentication for the G4F GUI (using any username and the API key as the password) and API key authentication for the API. This ensures that only authorized users can access both the interface and make API requests.

[Return to Home](/)