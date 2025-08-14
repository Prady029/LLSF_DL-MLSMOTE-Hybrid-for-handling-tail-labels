"""
Setup configuration for LLSF-DL MLSMOTE Python package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="llsf-dl-mlsmote",
    version="1.0.0",
    author="Prady029",
    author_email="",
    description="Hybrid approach combining LLSF-DL with MLSMOTE for handling tail labels in multi-label classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Prady029/LLSF_DL-MLSMOTE-Hybrid-for-handling-tail-labels",
    package_dir={"": "python_src"},
    packages=find_packages(where="python_src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0", 
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "isort>=5.9.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llsf-dl-mlsmote=evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="machine-learning multi-label classification imbalanced-learning tail-labels mlsmote llsf-dl",
    project_urls={
        "Bug Reports": "https://github.com/Prady029/LLSF_DL-MLSMOTE-Hybrid-for-handling-tail-labels/issues",
        "Source": "https://github.com/Prady029/LLSF_DL-MLSMOTE-Hybrid-for-handling-tail-labels",
        "Documentation": "https://github.com/Prady029/LLSF_DL-MLSMOTE-Hybrid-for-handling-tail-labels/blob/main/README.md",
    },
)
