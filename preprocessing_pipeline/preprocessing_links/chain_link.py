from abc import ABC, abstractmethod
from typing import Dict, Union


class ChainLink(ABC):

    def is_activated(self, link_config: Union[Dict[str, str], None]) -> bool:
        return link_config is not None and link_config.get('is_activated',True)

    @abstractmethod
    def run(self, global_config: Dict[str, str]):
        """
        Performs some operations related to data preprocessing
        """
