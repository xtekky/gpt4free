
# G4F Docker Setup

## Table of Contents
   - [Prerequisites](#prerequisites)
   - [Installation and Setup](#installation-and-setup)
   - [Testing the API](#testing-the-api)
   - [Troubleshooting](#troubleshooting)
   - [Stopping the Service](#stopping-the-service)


## Prerequisites
**Before you begin, ensure you have the following installed on your system:**
   - [Docker](https://docs.docker.com/get-docker/)
   - [Docker Compose](https://docs.docker.com/compose/install/)
   - Python 3.7 or higher
   - pip (Python package manager)

**Note:** If you encounter issues with Docker, you can run the project directly using Python.

## Installation and Setup

### Docker Method (Recommended)
1. **Clone the Repository**
   ```bash
   git clone https://github.com/xtekky/gpt4free.git
   cd gpt4free
   ```

2. **Build and Run with Docker Compose**

   Pull the latest image and run a container with Google Chrome support:
   ```bash
      docker pull hlohaus789/g4f
      docker-compose up -d
   ```
   Or run the small docker images without Google Chrome:
   ```bash
      docker-compose -f docker-compose-slim.yml up -d
   ```

3. **Access the API or the GUI**

   The api server will be accessible at `http://localhost:1337`

   And the gui at this url: `http://localhost:8080`

### Non-Docker Method
If you encounter issues with Docker, you can run the project directly using Python:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/xtekky/gpt4free.git
   cd gpt4free
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Server**
   ```bash
   python -m g4f.api.run
   ```

4. **Access the API or the GUI**

   The api server will be accessible at `http://localhost:1337`

   And the gui at this url: `http://localhost:8080`


## Testing the API
**You can test the API using curl or by creating a simple Python script:**
### Using curl
```bash
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "What is the capital of France?"}' http://localhost:1337/chat/completions
```

### Using Python
**Create a file named `test_g4f.py` with the following content:**
```python
import requests

url = "http://localhost:1337/v1/chat/completions"
body = {
    "model": "gpt-4o-mini", 
    "stream": False,
    "messages": [
        {"role": "assistant", "content": "What can you do?"}
    ]
}

json_response = requests.post(url, json=body).json().get('choices', [])

for choice in json_response:
    print(choice.get('message', {}).get('content', ''))
```

**Run the script:**
```bash
python test_g4f.py
```

## Troubleshooting
- If you encounter issues with Docker, try running the project directly using Python as described in the Non-Docker Method.
- Ensure that you have the necessary permissions to run Docker commands. You might need to use `sudo` or add your user to the `docker` group.
- If the server doesn't start, check the logs for any error messages and ensure all dependencies are correctly installed.

**_For more detailed information on API endpoints and usage, refer to the [G4F API documentation](docs/interference-api.md)._**



## Stopping the Service

### Docker Method
**To stop the Docker containers, use the following command:**
```bash
docker-compose down
```

### Non-Docker Method
If you're running the server directly with Python, you can stop it by pressing Ctrl+C in the terminal where it's running.

---

[Return to Home](/)
