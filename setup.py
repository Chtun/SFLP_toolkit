"""Package configuration and setup for spherical tiling and heatmap visualization."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Core package requirements
REQUIREMENTS = [
    "numpy==1.26.4",
    "pandas==2.2.2",
    "matplotlib>=3.4.0,<4.0.0",
    "seaborn>=0.11.0,<0.13.0",
    "ffmpeg-python>=0.2.0",
    "pyvista>=0.44.2",
    "Pillow>=9.0.0,<11.0.0",
    "opencv-python>=4.7.0,<4.10.0",
    "scipy>=1.7.0,<2.0.0",
    "scikit-learn>=1.0.0,<2.0.0",
    "pydantic",
    "pytube",
]

# CuPy (for GPU accelerated workflow)
CUPY_EXTRAS = [
    "cupy-cuda12x==13.5.1; sys_platform=='linux' or sys_platform=='darwin'",
    "cupy-cpu==13.5.1; sys_platform=='win32'",
]

# Testing
TESTING_REQUIREMENTS = [
    "pytest>=6.0.0,<7.0.0",
    "pytest-cov>=2.12.0,<3.0.0",
    "pytest-mock>=3.6.0,<4.0.0",
]

# Type checking
TYPE_CHECKING_REQUIREMENTS = [
    "mypy>=0.910,<1.0.0",
    "types-setuptools>=57.0.0,<58.0.0",
    "pandas-stubs>=1.2.0,<2.0.0",
    "typing-extensions>=3.10.0.0",
]

# Development requirements
DEV_REQUIREMENTS = [
    "black>=21.5b2",
    "isort>=5.9.0",
    "flake8>=3.9.0",
    "pylint>=2.8.0",
]

# Documentation requirements
DOCS_REQUIREMENTS = [
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=0.5.0",
    "myst-parser>=0.15.0",
]

setup(
    name="SFLP_toolkit",
    version="0.1.0",
    description="A system for encoding using the Spherical Fibonacci Lattice Pixelation method and evaluating against SOTA.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chitsein Htun",
    author_email="chtun@live.com",
    url="https://github.com/Chtun/SFLP_toolkit",
    packages=find_packages(exclude=["tests*", "examples*"]),
    
    # Core package dependencies
    install_requires=REQUIREMENTS,
    
    # Optional dependencies
    extras_require={
        "cuda": CUPY_EXTRAS,
        "testing": TESTING_REQUIREMENTS,
        "type_checking": TYPE_CHECKING_REQUIREMENTS,
        "dev": DEV_REQUIREMENTS,
        "docs": DOCS_REQUIREMENTS,
        "all": DEV_REQUIREMENTS + DOCS_REQUIREMENTS + CUPY_EXTRAS,
    },
    
    # Package metadata
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    
    # Package data
    include_package_data=True,
    package_data={
        "SFLP_toolkit": [
            "data/test_data/*.csv",
            "data/example_data/*.csv",
        ]
    },
    
    # Entry points for CLI
    entry_points={
        "console_scripts": [
            "SFLP_toolkit=SFLP_toolkit.cli:main",
        ]
    },
    
    keywords=[
        "fibonacci-lattice",
        "360-video",
        "encoding",
        "compression",
    ],
    
    platforms=["any"],
    license="MIT",
)
