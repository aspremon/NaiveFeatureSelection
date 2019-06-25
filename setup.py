from setuptools import find_packages, setup

DEPENDENCY_PACKAGE_NAMES = ["numpy","sklearn",]


def check_dependencies():
    missing_dependencies = []
    for package_name in DEPENDENCY_PACKAGE_NAMES:
        try:
            __import__(package_name)
        except ImportError:
            missing_dependencies.append(package_name)

    if missing_dependencies:
        raise ValueError(
            'Missing dependencies: {}. We recommend you follow '
            'the installation instructions at '
            'https://github.com/hassony2/manopth#installation'.format(
                missing_dependencies))


with open("README.md", "r") as fh:
    long_description = fh.read()

check_dependencies()

setup(
    name="NFS",
    version="0.0.1",
    author="Armin Askari, Alexandre d'Aspremont, Laurent El Ghaoui",
    author_email="aspremon@ens.fr",
    packages=find_packages(exclude=('tests',)),
    python_requires=">=3.5.0",
    description="Naive Feature Selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aspremon/NFS",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT",
        "Operating System :: OS Independent",
    ],
)