from setuptools import setup

setup(
    name='bayesian unet',
    version='1.0.0',
    install_requires=[
        'nibabel~=3.2.1',
        'tensorflow-gpu==1.15',
        'tensorflow-probability==0.7.0',
        'matplotlib==3.1.0',
        'colorcet==2.0.1',
        'brokenaxes==0.3.1',
        'pymongo==3.8.0',
        'numpy~=1.19.5',
        'tqdm~=4.62.3'
    ],
)