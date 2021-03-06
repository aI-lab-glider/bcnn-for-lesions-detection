from setuptools import setup

# Run with python setup_jon.py develop
setup(
    name='JON - bayesian unet CLi',
    version='1.0.0',
    install_requires=[
        'click'
    ],
    entry_point='''
        [console_scripts]
        jon=bayesian_cnn_prometheus.cli.entry_point:entry_point
    '''
)
