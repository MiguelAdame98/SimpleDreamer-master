from setuptools import setup, find_packages
from pathlib import Path


this_dir = Path(__file__).parent
setup(
    name="dreamer_mg",
    version="0.0.1",
    packages=find_packages(),         # finds dreamer_mg and sub-pkgs
    include_package_data=True,         # include *.yml configs
    install_requires=[
        # core
        "torch>=2.1",
        "numpy>=1.22",
        "pyyaml",
        "tqdm",
        "tensorboard",
        # environment deps (gym already required by MiniGrid fork)
        "gym>=0.17.0",
    ],
)