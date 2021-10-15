import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    description="A Hair Cell Analysis Toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=3',
    entry_points = {'console_scripts': ['hcat-segment=hcat.main:segment',
                                        'hcat-detect=hcat.main:detect',
                                        'hcat=hcat.main:cli']},
    install_requires = [
        'kornia>=0.5.2',
        'numpy>=1.10.0',
        'torch>=1.9.0',
        'matplotlib>=3.3.2',
        'scipy>=1.5.4',
        'scikit-image>=0.17.2',
        'tqdm>=4.60.0',
        'click>=8.0.0',
        'lz4>=3.1.3',
        'scikit-learn>=0.24.2',
        'GPy>=1.10.0',
        'torchvision>=0.10.0',
        'elasticdeform>=0.4.9',
        'wget>=3.2']
)