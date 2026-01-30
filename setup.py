"""
Setup configuration for OtterSearch
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ottersearch",
    version="1.0.0",
    author="OtterSearch Contributors",
    description="Lightweight image & PDF search with AI-powered semantic understanding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/khushwant18/OtterSearch",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Office/Business",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "flask>=2.3.0",
        "hnswlib>=0.8.0",
        "numpy>=1.23.0",
        "PyMuPDF>=1.23.0",
        "Pillow>=9.0.0",
        "clip>=1.0",
    ],
    entry_points={
        "console_scripts": [
            "ottersearch=ottersearch.cli:run_server",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["ui.html"],
    },
)
