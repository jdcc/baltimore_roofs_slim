from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="baltimoreroofs",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required_packages,
    entry_points={
        "console_scripts": [
            "roofs = baltimoreroofs.cli:roofs",
        ],
    },
)
