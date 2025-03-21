from abc import ABC
import torch
from recommender_core.utils.singleton import Singleton


class EmbeddingModelBase(Singleton, ABC):
    """
    Base for all embedding models.
    """

    def __init__(self, model_name: str, model_path: str| None = None, device: str| None = None,):
        self.device = device if device in ["cpu", "cuda"] else "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.model_name = model_name

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