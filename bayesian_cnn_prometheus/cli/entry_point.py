import click

from bayesian_cnn_prometheus.cli.evaluate import evaluate_with_config_scan_id
from bayesian_cnn_prometheus.cli.start_training import start_training


# from bayesian_cnn_prometheus.cli.analise import analise


@click.group()
def entry_point():
    """
    Entry point for Jon: Justifiable Oncology Nemesis
    """


entry_point.add_command(start_training)
entry_point.add_command(evaluate_with_config_scan_id)
# entry_point.add_command(analise)
