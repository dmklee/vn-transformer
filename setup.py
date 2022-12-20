import platform
from setuptools import setup, find_packages

setup(
    name='vn-transformer',
    version='0.0.1',
    description='Implementation of VectorNeuron-Transformer',
    license='MIT License',
    author='David Klee',
    author_email='klee.d@northeastern.edu',
    url='https://github.com/dmklee/vn-transformer',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>3.6.0',
    setup_requires='wheel',
    install_requires=[
        "numpy",
        "torch",
    ],
)
