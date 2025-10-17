#!/usr/bin/env python

# How to build source distribution
# python setup.py sdist --format bztar
# python setup.py sdist --format gztar
# python setup.py sdist --format zip


from setuptools import setup


VERSION = "0.1.0"


def setup_package():
    setup(
        name="miami_generator",
        version=VERSION,
        description="Creation of beautiful Miami plots",
        author="Louis-Philippe Lemieux Perreault",
        author_email="louis-philippe.lemieux.perreault@statgen.org",
        url="https://github.com/pgxcentre/manhattan_generator",
        license="MIT",
        entry_points={
            "console_scripts": ["miami_generator=miami_generator:main"],
        },
        py_modules=["miami_generator"],
        install_requires=[
            "matplotlib>=3.0",
            "polars>=0.19",
        ],
        classifiers=[
            "Intended Audience :: Science/Research",
            "License :: MIT",
            "Operating System :: Unix",
            "Operating System :: POSIX :: Linux",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
        ],
    )

    return


if __name__ == "__main__":
    setup_package()
