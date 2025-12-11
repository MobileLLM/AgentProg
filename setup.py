# setup.py

from setuptools import setup, find_packages

setup(
    name='agentprog',
    version='1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'agentprog=agentprog.cli:agentprog_cli'
        ],
    },
    install_requires=[
        'litellm', 
        'pandas',
        "structlog",
        "pillow",
        "python-dotenv",
        "requests",
        "tenacity",
        "rich",
        "textual",
        "pyfiglet",
        "google",
        "google-api-python-client",
        "ui-tars"
    ],
)
