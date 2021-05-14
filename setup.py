import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    description="Whole cochlea hair cell segmentation toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=3.9',
    entrypoints = {'console_scripts': ['hcat = hcat:analyze']}
)