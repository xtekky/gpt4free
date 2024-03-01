### G4F - Installation Guide

Follow these steps to install G4F from the source code:

1. **Clone the Repository:**

```bash
git clone https://github.com/xtekky/gpt4free.git
```

2. **Navigate to the Project Directory:**

```bash
cd gpt4free
```

3. **(Optional) Create a Python Virtual Environment:**

It's recommended to isolate your project dependencies. You can follow the [Python official documentation](https://docs.python.org/3/tutorial/venv.html) for virtual environments.

```bash
python3 -m venv venv
```

4. **Activate the Virtual Environment:**

- On Windows:

```bash
.\venv\Scripts\activate
```

- On macOS and Linux:

```bash
source venv/bin/activate
```

5. **Install Minimum Requirements:**

Install the minimum required packages:

```bash
pip install -r requirements-min.txt
```

6. **Or Install All Packages from `requirements.txt`:**

If you prefer, you can install all packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

7. **Start Using the Repository:**

You can now create Python scripts and utilize the G4F functionalities. Here's a basic example:

Create a `test.py` file in the root folder and start using the repository:

```python
import g4f
# Your code here
```

[Return to Home](/)