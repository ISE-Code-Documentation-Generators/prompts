import setuptools
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = "0.0.1"
DESCRIPTION = "To be added in the future"


llama_dependencies = [
    "llama-index-program-openai==0.1.4",
    "llama-index-llms-llama-api==0.1.4",
    "llama-index==0.10.23",
]

dolly_dependencies = [
    "accelerate==0.28.0",
    "transformers==4.39.1",
]

setuptools.setup(
    name="ise_cdg_prompts",
    version=VERSION,
    author="Sepehr Kianian",
    author_email="sepehr.kianian.r@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "ise_cdg_data @ git+https://github.com/ISE-Code-Documentation-Generators/data.git",
        "ise_cdg_utility @ git+https://github.com/ISE-Code-Documentation-Generators/utility.git",
    ]
    + llama_dependencies
    + dolly_dependencies,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
)
