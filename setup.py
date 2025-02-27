from setuptools import setup, find_packages

setup(
    name="candle-indicators",
    version="0.1.0",
    author="Bwahharharrr",
    description="Technical analysis indicators for financial data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Bwahharharrr/candle-indicators",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.18.0",
    ],
) 