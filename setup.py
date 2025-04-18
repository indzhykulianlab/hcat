import os
from wheel.bdist_wheel import bdist_wheel
import glob
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py

from Cython.Build import cythonize

EXCLUDE_FILES = ["resources.py", "adjust.py"]


def get_ext_paths(root_dir, exclude_files):
    """get filepaths for compilation"""
    paths = []

    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            # print(f'{filename=}, {os.path.splitext(filename)[1]}, {os.path.splitext(filename)[1] == "resources.py"}')
            if os.path.splitext(filename)[1] != ".py":
                continue
            elif filename in EXCLUDE_FILES:
                continue


            file_path = os.path.join(root, filename)
            if file_path in exclude_files:
                continue

            paths.append(file_path)
    # print(paths)
    # raise ValueError
    return paths


# noinspection PyPep8Naming
class build_py(_build_py):
    def find_package_modules(self, package, package_dir):
        # ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
        modules = super().find_package_modules(package, package_dir)
        filtered_modules = []
        for pkg, mod, filepath in modules:
            if os.path.exists(filepath.replace(".py", ".c")):
                continue
            filtered_modules.append(
                (
                    pkg,
                    mod,
                    filepath,
                )
            )
        # print(f'{filtered_modules=}')
        return filtered_modules


class CommandBdistWheel(bdist_wheel):
    # Called almost exactly before filling `.whl` archive
    def write_wheelfile(self, *args, **kwargs):
        dr = f"{self.bdist_dir}/hcat"
        paths = [
            path
            for path in glob.glob(f"{dr}/**/*.py", recursive=True)
            if os.path.basename(path) not in ["resources.py", "adjust.py"]
        ]
        for path in paths:
            os.remove(path)
        super().write_wheelfile(*args, **kwargs)


requirements = [
    "pyside6>=6.4.1",
    "torch>=2.0.0",
    "py-machineid>=0.2.1",
    "torchvision>=0.14.1",
    "wget>=3.2",
    "scikit-image>=0.19.3",
    "numpy<2",
    "tqdm>=4.64.1",
    "yacs>=0.1.8",
    "numba>=0.54.0",
    "Cython>=3.0.0b3",
    "requests",
    "imagecodecs"
]

setup(
    name="hcat",
    version="2.0.4",
    packages=find_packages(),
    # ext_modules=cythonize(
    #     get_ext_paths("hcat", EXCLUDE_FILES),
    #     compiler_directives={
    #         "language_level": 3,
    #         "always_allow_keywords": True,
    #         "annotation_typing": False,
    #     },
    # ),
    # cmdclass={"build_py": build_py, "bdist_wheel": CommandBdistWheel},
    entry_points={"console_scripts": ["hcat = hcat.__main__:launch"]},
    install_requires=requirements,
)
