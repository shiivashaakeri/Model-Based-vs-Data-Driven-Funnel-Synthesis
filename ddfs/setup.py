"""
Setup script for DDFS package.

This file is maintained for backward compatibility with older pip versions
and editable installs. The main configuration is in pyproject.toml.
"""

from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        packages=find_packages(where="ddfs", include=["ddfs", "ddfs.*"]),
        package_dir={"": "ddfs"},
    )

