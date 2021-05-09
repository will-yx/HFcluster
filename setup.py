from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["seaborn", "plotly", "notebook", "pandas", "numpy", "scikit-learn", "scipy", "scanpy", "louvain", "leidenalg", "kneed", "dill"]

setup(
    name="HFcluster",
    version="0.0.1",
    author="Yu Xin (Will) Wang",
    author_email="willw1@stanford.edu",
    description="Clustering and analysis tools for single cell spatial data from multiplex imaging",
    long_description=readme,
    long_description_content_type="clustering/single cell analysis",
    url="https://github.com/will-yx/HFcluster/",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)