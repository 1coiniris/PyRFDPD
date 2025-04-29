from setuptools import setup, find_packages
from pyrfdpd import __version__
from docutils import core
from pathlib import Path

path = Path.cwd()
# parse description section text
with open(str(path / "README.rst"), "r") as f:
    data = f.read()
    readme_nodes = list(core.publish_doctree(data))
    for node in readme_nodes:
        if node.astext().startswith("Description"):
            long_description = node.astext().rsplit("\n\n")[1]

# parse package requirements from text file
with open(str(path / "requirements.txt"), "r") as f:
    req_list = f.read().split("\n")

setup(
    name="pyrfdpd",
    version=__version__,
    author="Zhe Li",
    author_email="ataraxialex@gmail.com",
    description="Python package for radio frequency digital predistortion techniques",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/SEU-MSLab/pyrfdpd",
    packages=find_packages(),
    license="MIT",
    keywords="communication DPD pytroch volterra",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=req_list,
    include_package_data=True,
    zip_safe=False,
)
