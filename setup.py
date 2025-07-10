import os

from setuptools import find_packages, setup

current_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(current_dir, 'README.md'), "r", encoding="utf-8") as f:
    long_description = f.read()

long_description = long_description.replace("[!NOTE]", "")

INSTALL_REQUIRE = [
    "requests",
    "aiohttp",
    "brotli",
    "pycryptodome",
    "nest_asyncio",
]

EXTRA_REQUIRE = {
    'all': [
        "curl_cffi>=0.6.2",
        "certifi",
        "browser_cookie3",         # get_cookies
        "duckduckgo-search>=5.0",  # internet.search
        "beautifulsoup4",          # internet.search and bing.create_images
        "platformdirs",
        "aiohttp_socks",           # proxy
        "pillow",                  # image
        "cairosvg",                # svg image
        "werkzeug", "flask",       # gui
        "fastapi",                 # api
        "uvicorn",                 # api
        "nodriver",
        "python-multipart",
        "a2wsgi",
        "pywebview",
        "plyer",
        "setuptools",
        "markitdown[all]",
        "python-dotenv"
    ],
    'slim': [
        "curl_cffi>=0.6.2",
        "certifi",
        "browser_cookie3",
        "duckduckgo-search>=5.0"  ,# internet.search
        "beautifulsoup4",          # internet.search and bing.create_images
        "aiohttp_socks",           # proxy
        "pillow",                  # image
        "werkzeug", "flask",       # gui
        "fastapi",                 # api
        "uvicorn",                 # api
        "python-multipart",
        "a2wsgi",
        "markitdown[all]",
        "python-dotenv"
    ],
    "image": [
        "pillow",
        "cairosvg",
        "beautifulsoup4"
    ],
    "webview": [
        "pywebview",
        "platformdirs",
        "plyer",
        "cryptography",
    ],
    "api": [
        "loguru", "fastapi",
        "uvicorn",
        "python-multipart",
    ],
    "gui": [
        "werkzeug", "flask",
        "beautifulsoup4", "pillow",
        "duckduckgo-search>=5.0",
    ],
    "search": [
        "beautifulsoup4",
        "pillow",
        "duckduckgo-search>=5.0",
    ],
    "local": [
        "gpt4all"
    ],
    "files": [
        "beautifulsoup4",
        "markitdown[all]"
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
        'g4f': []
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
        "gpt4free",
        "gpt4free.js",
        "g4f",
        "g4f.dev",
        "javascript",
        "npm",
        "browser",
        "gpt",
        "chatgpt",
        "deepseek",
        "openai",
        "ai",
        "client",
        "sdk",
        "free",
        "ai",
        "gpt-4",
        "gpt-4o",
        "chat",
        "api",
        "browser",
        "ai",
        "ai",
        "js",
        "client",
        "text",
        "generation",
        "image",
        "generation",
        "in-browser",
        "ai",
        "frontend",
        "ai",
        "openai",
        "alternative",
        "javascript",
        "ai",
        "library",
        "nodejs",
        "prompt",
        "engineering",
        "chatbot",
        "ai",
        "integration"
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
