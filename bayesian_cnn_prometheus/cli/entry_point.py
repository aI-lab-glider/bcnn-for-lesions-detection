import click

from bayesian_cnn_prometheus.cli.start_training import start_training


@click.group()
def entry_point():
    """
    Entry point for Jon: Justifiable Oncology Nemesis
    """


entry_point.add_command(start_training)
