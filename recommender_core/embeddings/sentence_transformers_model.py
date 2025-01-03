from sentence_transformers import SentenceTransformer
from torch import Tensor

from recommender_core.embeddings.base import BaseEmbeddingModel


class SentenceTransformerModel(BaseEmbeddingModel):
    def __init__(self, model_name: str, device=None):
        super().__init__(device)
        self._model = SentenceTransformer(model_name)

    def encode(self, sentences: str | list[str], **kwargs) -> Tensor:
        return self._model.encode(
            sentences,
            device=self.device,
            normalize_embeddings=True,
            convert_to_tensor=True
        )