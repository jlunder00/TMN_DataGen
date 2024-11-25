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
    version='0.6.2',
    description="Tree Matching Network Data Generator",
    author="toast",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        'stanza': ['stanza>=1.2.3'],
        'all': [
            'stanza>=1.2.3',
            'regex>=2022.1.18',
            'unicodedata2>=15.0.0'
        ]
    },
    include_package_data=True,
    zip_safe=False,
)
