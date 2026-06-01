from setuptools import setup, find_packages

setup(
    name="traverse",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.9.0",
        "tensorflow>=2.5.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.20.0",
        "pydantic>=1.10.0",
    ],
    entry_points={
        "console_scripts": [
            "traverse=traverse.cli.main:main",
        ],
    },
)
