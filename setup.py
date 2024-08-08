from setuptools import find_packages, setup

setup(
    name="rl_rht",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    setup_requires=["setuptools_scm"],
    install_requires=[
        "pandas>=2.0.0",
        # https://stackoverflow.com/questions/66060487/valueerror-numpy-ndarray-size-changed-may-indicate-binary-incompatibility-exp
        "numpy>=1.20.0",
        "typer>=0.7.0",
        "scipy",
        "matplotlib",
        "seaborn",
        "einops",
        # https://stackoverflow.com/questions/54364289/install-failure-for-pytables-in-terminal
        "tables",
        "torch",
        "pyarrow",
        "fastparquet",
    ],
    extras_require={
        "dev": [  # https://github.com/jazzband/pip-tools/issues/1617
            "pip-tools>=6.6.1",
            "black>=22.12.0",
            "mypy",
            "ipykernel",
        ]
    },
)
