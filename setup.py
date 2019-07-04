from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="naive_feature_selection",
    version="0.0.1",
    author="Armin Askari, Alexandre d'Aspremont, Laurent El Ghaoui",
    author_email="aspremon@ens.fr",
    packages=find_packages(),
    python_requires=">=3.5.0",
    install_requires=["numpy"],
    description="Naive Feature Selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aspremon/NaiveFeatureSelection",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT",
        "Operating System :: OS Independent",
    ],
)
