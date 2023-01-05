from setuptools import find_packages, setup
# print(REQUIREMENTS)

setup(
    name='stylometry_utils',
    packages=find_packages(include=['stylometry_utils']),
    version='0.1.3',
    description='Collection of functions and utilities to run stylometry experiments',
    author='Simone Martin Marotta',
    install_requires=[
        "keras",
        "nltk",
        "numpy",
        "openpyxl",
        "pandas",
        "scikit-learn",
        "scipy",
        "tensorboard",
        "tensorboard-data-server",
        "tensorboard-plugin-wit",
        "tensorflow",
        "tensorflow-estimator",
        "tensorflow-intel",
        "tensorflow-io-gcs-filesystem",
        "tokenizers",
        "tornado",
        "tqdm",
        "transformers",
        "typing_extensions",
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)