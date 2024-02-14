import codecs
import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, 'README.md'), encoding='utf-8') as fh:
    long_description = '\n' + fh.read()

INSTALL_REQUIRE = [
    "requests",
    "aiohttp",
]

EXTRA_REQUIRE = {
    'all': [
        "curl_cffi>=0.6.0b9",
        "certifi",
        "async-property",          # openai
        "py-arkose-generator",     # openai
        "browser_cookie3",         # get_cookies
        "PyExecJS",                # GptForLove
        "duckduckgo-search",       # internet.search
        "beautifulsoup4",          # internet.search and bing.create_images
        "brotli",                  # openai
        "platformdirs",            # webdriver
        "undetected-chromedriver", # webdriver
        "setuptools",              # webdriver
        "aiohttp_socks",           # proxy
        "pillow",                  # image
        "cairosvg",                # svg image
        "werkzeug", "flask",       # gui
        "loguru", "fastapi",
        "uvicorn", "nest_asyncio", # api
    ],
    "image": [
        "pillow",
        "cairosvg",
        "beautifulsoup4"
    ],
    "webdriver": [
        "platformdirs",
        "undetected-chromedriver",
        "setuptools"
    ],
    "openai": [
        "async-property",
        "py-arkose-generator",
        "brotli"
    ],
    "api": [
        "loguru", "fastapi",
        "uvicorn", "nest_asyncio"
    ],
    "gui": [
        "werkzeug", "flask",
        "beautifulsoup4", "pillow",
        "duckduckgo-search",
        "browser_cookie3"
    ]
}

DESCRIPTION = (
    'The official gpt4free repository | various collection of powerful language models'
)

# Setting up
setup(
    name='g4f',
    version=os.environ.get("G4F_VERSION"),
    author='Tekky',
    author_email='<support@g4f.ai>',
    description=DESCRIPTION,
    long_description_content_type='text/markdown',
    long_description=long_description,
    packages=find_packages(),
    package_data={
        'g4f': ['g4f/interference/*', 'g4f/gui/client/*', 'g4f/gui/server/*', 'g4f/Provider/npm/*']
    },
    include_package_data=True,
    install_requires=INSTALL_REQUIRE,
    extras_require=EXTRA_REQUIRE,
    entry_points={
        'console_scripts': ['g4f=g4f.cli:main'],
    },
    url='https://github.com/xtekky/gpt4free',  # Link to your GitHub repository
    project_urls={
        'Source Code': 'https://github.com/xtekky/gpt4free',  # GitHub link
        'Bug Tracker': 'https://github.com/xtekky/gpt4free/issues',  # Link to issue tracker
    },
    keywords=[
        'python',
        'chatbot',
        'reverse-engineering',
        'openai',
        'chatbots',
        'gpt',
        'language-model',
        'gpt-3',
        'gpt3',
        'openai-api',
        'gpt-4',
        'gpt4',
        'chatgpt',
        'chatgpt-api',
        'openai-chatgpt',
        'chatgpt-free',
        'chatgpt-4',
        'chatgpt4',
        'chatgpt4-api',
        'free',
        'free-gpt',
        'gpt4free',
        'g4f',
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Operating System :: Unix',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
    ],
)
