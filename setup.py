from setuptools import setup, find_packages

setup(
    name="acsefunctions",
    version="0.1.0",
    description="A numerical package.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/acsefunctions_repo",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
