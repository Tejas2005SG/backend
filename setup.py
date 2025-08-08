from setuptools import setup, find_packages

setup(
    name="backend",
    version="0.1.0",
    packages=find_packages(where="backend"),
    package_dir={"": "backend"},
    install_requires=[
        "fastapi>=0.116.1",
        "google-generativeai>=0.8.5",
        "huggingface-hub>=0.34.4",
        "langchain>=0.3.27",
        "langchain-community>=0.3.27",
        "langchain-huggingface>=0.3.1",
        "langchain-pinecone>=0.2.11",
        "pinecone>=5.0.0",
        "prisma>=0.15.0",
        "pydantic>=2.11.7",
        "pypdf>=5.9.0",
        "python-dotenv>=1.1.1",
        "requests>=2.32.4",
        "sentence-transformers>=5.1.0",
        "uvicorn>=0.35.0",
    ],
    python_requires=">=3.11",
)
