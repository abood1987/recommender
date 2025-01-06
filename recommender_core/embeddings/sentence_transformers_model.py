import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
from torch import Tensor
from recommender_core.embeddings.base import BaseEmbeddingModel


class SentenceTransformerModel(BaseEmbeddingModel):
    def __init__(self, model_name: str, model_path: str| None = None, device: str| None =None):
        super().__init__(device)
        self.model_path  = model_path
        self.model_name  = model_name
        self._model = self._get_model()

    def _get_model(self):
        if not self.model_path:
            return SentenceTransformer(self.model_name)
        path = Path(self.model_path)
        if path.exists() and any(path.iterdir()):
            return SentenceTransformer(self.model_path)
        else:
            path.mkdir(parents=True, exist_ok=True)
            model = SentenceTransformer(self.model_path)
            model.save(self.model_path)
            return model

    def encode(self, sentences: str | list[str], **kwargs) -> Tensor:
        return self._model.encode(
            sentences,
            device=self.device,
            normalize_embeddings=True,
            convert_to_tensor=True
        )