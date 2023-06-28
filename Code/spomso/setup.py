from setuptools import setup, find_packages

VERSION = '1.0.0'
DESCRIPTION = 'SPOMSO'
LONG_DESCRIPTION = 'Python package for generating geometry in two or three dimensions using SDFs.'

# Setting up
setup(
    name="SPOMSO",
    version=VERSION,
    author="Peter Ropač",
    author_email="<peter.ropac@fmf.uni-lj.si>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url='https://github.com/peterropac/SPOMSO/tree/master/Scripts',
    packages=find_packages(),
    install_requires=['numpy', 'scipy'],
    keywords=['python', 'geometry'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Linux :: Ubuntu",
        "Operating System :: Microsoft :: Windows",
    ]
)