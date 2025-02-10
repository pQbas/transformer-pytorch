import os
from setuptools import setup, find_packages

# Function to read the requirements from requirements.txt
def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(os.path.join(os.path.dirname(__file__), filename), 'r') as req_file:
        return req_file.read().splitlines()

# Short description of your project
description_ = """Library that gives an abstraction layer to work with Alexnet CNN"""

setup(
    name='transformer',  # Name of your library
    version='0.1.0',  # Initial version number
    author='Percy Cubas',  # Your name
    author_email='pcubasm1@gmail.com',  # Your email address
    url='https://github.com/pQbas/transformer-pytorch.git',  # URL of your project's repository
    packages=find_packages(),  # Automatically find packages in the directory
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Change as per your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # Specify the minimum Python version required 
)
