# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:21:37 2025

@author: yzhao
"""

from setuptools import setup, find_packages


with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="r_peak_detection",
    version="0.0.1",
    author="Yue Zhao",
    author_email="yuezhao@rochester.edu",
    description="A command-line tool to label R-peaks based on ECG data.",
    packages=find_packages(),
    py_modules=["detect_r_peaks"],
    install_requires=required,
    entry_points={
        "console_scripts": [
            "detect-r-peaks=detect_r_peaks:main",
        ],
    },
    python_requires=">=3.9",
)
