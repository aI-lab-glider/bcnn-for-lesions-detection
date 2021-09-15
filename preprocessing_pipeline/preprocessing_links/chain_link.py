from abc import ABC, abstractmethod
from typing import Union


class ChainLink(ABC):

    def is_activated(self, link_config: Union[dict[str, str], None]) -> bool:
        return link_config is not None and getattr(link_config, 'is_activated', True)

    @abstractmethod
    def run(self, global_config: dict[str, str]):
        """
        Performs some operations related to data preprocessing
        """
