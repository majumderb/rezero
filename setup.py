import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='rezero',
    version='0.1.0',
    author="Thomas Bachlechner, Bodhisattwa Prasad Majumder, Huanru Henry Mao, Garrison W. Cottrell, Julian McAuley",
    author_email="henry@calclavia.com",
    description="ReZero networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/majumderb/rezero",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
