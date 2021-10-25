import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="gradients",
    version="0.0.3",
    author="Saranraj Nambusubramaniyan",
    author_email="saran_nns@hotmail.com",
    description="Gradient Checker for Custom built PyTorch Models",
    license="OSI Approved :: MIT License",
    keywords="""PyTorch,Artificial Neural Networks,Gradients,
                  BackPropagation, Machine Learning""",
    url="https://github.com/Saran-nns/gradients",
    packages=["gradients"],
    data_files=["LICENSE"],
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    include_package_data=True,
    install_requires=["numpy"],
    zip_safe=False,
)

