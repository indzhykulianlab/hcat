import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    description="Whole cochlea hair cell segmentation toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=3',
    entry_points = {'console_scripts': ['hcat = hcat.main:analyze']},
    install_requires = [
        'kornia>=0.5.2',
        'numpy>=1.10.0',
        'torch>=1.8.1',
        'matplotlib>=3.3.2',
        'scipy>=1.5.4',
        'scikit-image>=0.17.2',
        'tqdm>=4.60.0',
        'click>=8.0.0',
        'lz4>=3.1.3',
        'scikit-learn>=0.24.2',
        'GPy>=1.10.0',
        'torchvision>=0.9.1',
        'elasticdeform>=0.4.9']
)