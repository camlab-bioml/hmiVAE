from setuptools import find_packages, setup

setup(
    name="hmiVAE",
    version="0.1.0",
    url="https://github.com/camlab-bioml/hmiVAE",
    project_urls={
        "Issues": "https://github.com/camlab-bioml/hmiVAE/issues",
        "Source": "https://github.com/camlab-bioml/hmiVAE",
    },
    author="Shanza Ayub",
    author_email="sayub@lunenfeld.ca",
    packages=find_packages(),
    package_dir={"hmiVAE": "hmiVAE"},
    package_data={"": ["*.json", "*.html", "*.css"]},
    include_package_data=True,
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords=["imaging mass cytometry variational autoencoder"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
    ],
    license="See License",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
        "pytorch-lightning",
        "scanpy",
        "anndata",
        "wandb",
        "typing"
    ],
    python_requires=">=3.8.0",
)