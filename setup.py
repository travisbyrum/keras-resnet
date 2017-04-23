import setuptools

setuptools.setup(
    author="Allen Goodman",
    author_email="allen.goodman@icloud.com",
    install_requires=[
        "keras"
    ],
    license="MIT",
    name="keras-resnet",
    package_data={
        "keras-resnet": [
            "data/checkpoints/*/*.hdf5",
            "data/logs/*/*.csv",
            "data/notebooks/*/*.ipynb"
        ]
    },
    packages=setuptools.find_packages(
        exclude=[
            "tests"
        ]
    ),
    url="https://github.com/0x00b1/keras-resnet",
    version="0.0.1"
)