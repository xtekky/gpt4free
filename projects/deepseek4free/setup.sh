git clone https://github.com/xtekky/deepseek4free.git

echo "recursive-include dsk/wasm *" >> ./deepseek4free/MANIFEST.in

echo "from setuptools import find_packages, setup

INSTALL_REQUIRE = [
    'curl-cffi',
    'wasmtime',
    'numpy',
]

DESCRIPTION = (
    'The official deepseek4free repository | various collection of powerful language models'
)

# Setting up
setup(
    name='dsk',
    version='0.0.1.0',
    author='Tekky',
    author_email='<support@g4f.ai>',
    description=DESCRIPTION,
    long_description_content_type='text/markdown',
    long_description='',
    packages=find_packages(),
    include_package_data=True,
    install_requires=INSTALL_REQUIRE
)" >> ./deepseek4free/setup.py

pip install ./deepseek4free --break-system-packages