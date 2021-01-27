import setuptools
from distutils.core import setup

with open("../readme.md", "r") as fh:
    long_description = fh.read()

setup(
    name="bifrost", 
    version="0.0.16",
    author="Axel Stjerngren",
    author_email="axelstjerngren@protonmail.com",
    description="A package for TVM and STONNE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/axelstjerngren/level-4-project",
    packages=setuptools.find_packages(),
    package_data={'': ['lib/*.so']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires = [
        "torch",
        "pytest",
        "pillow"
    ]

)