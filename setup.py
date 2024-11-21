from setuptools import setup, find_packages
from pathlib import Path


def load_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()

long_description = (Path(__file__).parent / "README.md").read_text()

required_core = load_requirements("requirements.txt")
required_datasci = load_requirements("requirements-datasci.txt")
required_aws = load_requirements("requirements-aws.txt")
required_dev = load_requirements("requirements-dev.txt")
required_dl = load_requirements("requirements-dl.txt")

required_all = required_datasci + required_aws + required_dev + required_dl

setup(
    name="pytorch_huggingface_examples",
    version=1.0,
    description="This is me trying to learn more about Hugging Face and PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Matthias Seeger",
    packages=find_packages(
        include=[
            "huggingface_utils",
            "huggingface.*",
        ]
    ),
    extras_require={
        "datasci": required_datasci,
        "dl": required_dl,
        "aws": required_aws,
        "dev": required_dev,
        "all": required_all,
    },
    install_requires=required_core,
    include_package_data=True,
)
