import codecs
import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

with open("etc/interference/requirements.txt") as f:
    api_required = f.read().splitlines()

VERSION = '0.1.5.7'
DESCRIPTION = (
    "The official gpt4free repository | various collection of powerful language models"
)

# Setting up
setup(
    name="g4f",
    version=VERSION,
    author="Tekky",
    author_email="<support@g4f.ai>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    package_data={"g4f": ["g4f/gui/client/*", "g4f/gui/server/*"]},
    include_package_data=True,
    data_files=["etc/interference/app.py"],
    install_requires=required,
    extras_require={"api": api_required},
    entry_points={
        "console_scripts": ["g4f=interference.app:main"],
    },
    url="https://github.com/xtekky/gpt4free",  # Link to your GitHub repository
    project_urls={
        "Source Code": "https://github.com/xtekky/gpt4free",  # GitHub link
        "Bug Tracker": "https://github.com/xtekky/gpt4free/issues",  # Link to issue tracker
    },
    keywords=[
        "python",
        "chatbot",
        "reverse-engineering",
        "openai",
        "chatbots",
        "gpt",
        "language-model",
        "gpt-3",
        "gpt3",
        "openai-api",
        "gpt-4",
        "gpt4",
        "chatgpt",
        "chatgpt-api",
        "openai-chatgpt",
        "chatgpt-free",
        "chatgpt-4",
        "chatgpt4",
        "chatgpt4-api",
        "free",
        "free-gpt",
        "gpt4free",
        "g4f",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)