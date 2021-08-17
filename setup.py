from setuptools import setup
with open("README.md") as fh:
    long_description = fh.read()

# def local_scheme(version):
#     '''
#     To aviod version error in uploading to testPyPI.
#     '''
#     return ""

setup(
    name="arginfer",
    version ='0.0.1',
    author="Ali Mahmoudi",
    author_email="alimahmoodi29@gmail.com",
    description="Infer the Ancestral Recombination Graph",
    long_description=long_description,
    packages=["arginfer"],
    long_description_content_type="text/markdown",
    url="http://pypi.python.org/pypi/arginfer",
    python_requires=">=3.4",
    entry_points={
        "console_scripts": [
            "arginfer=arginfer.__main__:main",
        ]
    },
    setup_requires=["setuptools_scm"],
    install_requires=[
        "msprime",
        "numpy",
        "pandas",
        "bintrees",
        "tqdm",
        "sortedcontainers"
    ],
    project_urls={
        "Source": "https://github.com/alimahmoudi29/arginfer",
        "Bug Reports": "https://github.com/alimahmoudi29/arginfer/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Development Status :: 3 - Alpha" # 4 - Beta 5 - Production/Stable
    ],
    # use_scm_version={"local_scheme": local_scheme,
    #                  "write_to": "arginfer/_version.py"},
)
