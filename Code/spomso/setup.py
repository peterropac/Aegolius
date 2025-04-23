from setuptools import setup, find_packages

VERSION = '1.3.0'
DESCRIPTION = 'SPOMSO'
LONG_DESCRIPTION = 'Python package for generating geometry with SDFs.'

# Setting up
setup(
    name="SPOMSO",
    version=VERSION,
    author="Peter Ropač",
    author_email="<peter.ropac@fmf.uni-lj.si>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url='https://github.com/peterropac/Aegolius',
    packages=find_packages(),
    install_requires=['numpy', 'scipy'],
    extras_require={"autodiff": ["jax", "jaxlib"],
                    "plot": ["matplotlib", "plotly"],
                    "all": ["jax", "jaxlib", "matplotlib", "plotly"]},
    keywords=['python', 'geometry'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)