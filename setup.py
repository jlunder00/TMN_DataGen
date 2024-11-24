#setup.py
import os
from setuptools import setup, find_packages

requirements = []
with open(os.path.dirname(__file__) + "/requirements.txt", "r") as R:
    for line in R:
        package = line.strip()
        requirements.append(package)

setup(
    name="TMN_DataGen",
    version='0.3.10',
    description="Tree Matching Network Data Generator",
    author="toast",
    packages=find_packages(),
    install_requires=requirements,
    zip_safe=False,
)
