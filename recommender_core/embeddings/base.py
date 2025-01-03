from abc import ABC

import torch

from recommender_core.utils.singleton import Singleton


class BaseEmbeddingModel(Singleton, ABC):
    """
    Base for all the embedding models.
    """

    def __init__(self, device: str| None = None,):
        self.device = (
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )

    def size(self) -> int:
        """
        :return: size of the embedding.
        """
        raise NotImplementedError

    def encode(self, sentences: str | list[str], **kwargs) -> list[float]:
        """
        :param sentences: text to embed.
        :return: text embedding.
        """
        raise NotImplementedError