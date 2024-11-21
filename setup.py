import os
from setuptools import setup, find_packages

requirements = []
with open(os.path.dirname(__file__) + "/requirements.txt", "r") as R:
    for line in R:
        package = line.strip()
        requirements.append(package)

setup(
    name="TMN_DataGen",
    version = '0.0.0',
    description="Tools to convert text into lemma dependency trees formatted for TMN training",
    url="git@github.com:jlunder00/TMN_DataGen.git",
    author="Jason Lunder",
    packages=find_packages(),
    install_requires=requirements,
    zip_safe=False,
)
