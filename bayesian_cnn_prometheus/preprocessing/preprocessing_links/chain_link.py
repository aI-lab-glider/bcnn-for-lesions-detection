from abc import abstractmethod
from typing import Dict, Union


class ChainLink:
    def __init__(self, global_config: dict):
        self.global_config = global_config

    @staticmethod
    def is_activated(link_config: Union[Dict[str, str], None]) -> bool:
        return link_config is not None and link_config.get('is_activated', True)

    @abstractmethod
    def run(self, global_config: Dict[str, str]):
        """
        Performs some operations related to data preprocessing
        """