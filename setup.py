import subprocess

from setuptools import find_packages, setup


def get_version() -> str:
    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().lstrip("v")
        return tag
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "0.0.0"

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="blaze-torch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=get_version(),
    description="A PyTorch adapter for forward-only model definition",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(include=["blaze*"]),
    install_requires=["torch>=2.0"],
    extras_require={"dev": ["pytest>=8.0"]},
)
