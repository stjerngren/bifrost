import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="abacus", 
    version="0.0.1",
    author="Axel Stjerngren",
    author_email="axelstjerngren@protonmail.com",
    description="A package for TVM and STONNE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/axelstjerngren/level-4-project",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)