
# G4F - Git Installation Guide

This guide provides step-by-step instructions for installing G4F from the source code using Git.


## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
   1. [Clone the Repository](#1-clone-the-repository)
   2. [Navigate to the Project Directory](#2-navigate-to-the-project-directory)
   3. [Set Up a Python Virtual Environment](#3-set-up-a-python-virtual-environment-recommended)
   4. [Activate the Virtual Environment](#4-activate-the-virtual-environment)
   5. [Install Dependencies](#5-install-dependencies)
   6. [Verify Installation](#6-verify-installation)
3. [Usage](#usage)
4. [Troubleshooting](#troubleshooting)
5. [Additional Resources](#additional-resources)

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:
- Git
- Python 3.7 or higher
- pip (Python package installer)

## Installation Steps

### 1. Clone the Repository
**Open your terminal and run the following command to clone the G4F repository:**
```bash
git clone https://github.com/xtekky/gpt4free.git
```

### 2. Navigate to the Project Directory
**Change to the project directory:**
```bash
cd gpt4free
```

### 3. Set Up a Python Virtual Environment (Recommended)
**It's best practice to use a virtual environment to manage project dependencies:**
```bash
python3 -m venv venv
```

### 4. Activate the Virtual Environment
**Activate the virtual environment based on your operating system:**
- **Windows:**
  ```bash
  .\venv\Scripts\activate
  ```

- **macOS and Linux:**
  ```bash
  source venv/bin/activate
  ```

### 5. Install Dependencies
**You have two options for installing dependencies:**

#### Option A: Install Minimum Requirements
**For a lightweight installation, use:**
```bash
pip install -r requirements-min.txt
```

#### Option B: Install All Packages
**For a full installation with all features, use:**
```bash
pip install -r requirements.txt
```

### 6. Verify Installation
You can now create Python scripts and utilize the G4F functionalities. Here's a basic example:

**Create a `g4f-test.py` file in the root folder and start using the repository:**
```python
import g4f
# Your code here
```

## Usage
**After installation, you can start using G4F in your Python scripts. Here's a basic example:**
```python
import g4f

# Your G4F code here
# For example:
from g4f.client import Client

client = Client()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Say this is a test"
        }
    ]
    # Add any other necessary parameters
)

print(response.choices[0].message.content)
```

## Troubleshooting
**If you encounter any issues during installation or usage:**
   1. Ensure all prerequisites are correctly installed.
   2. Check that you're in the correct directory and the virtual environment is activated.
   3. Try reinstalling the dependencies.
   4. Consult the [G4F documentation](https://github.com/xtekky/gpt4free) for more detailed information.

## Additional Resources
   - [G4F GitHub Repository](https://github.com/xtekky/gpt4free)
   - [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
   - [pip Documentation](https://pip.pypa.io/en/stable/)

---

**_For more information or support, please visit the [G4F GitHub Issues page](https://github.com/xtekky/gpt4free/issues)._**


---  
[Return to Home](/)
