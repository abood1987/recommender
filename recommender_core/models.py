from django.db import models
from pgvector.django import VectorField, CosineDistance

from recommender.settings import VECTOR_SETTINGS
from recommender_core.utils.helper import get_embedding_model


class BaseVectorModel(models.Model):
    embedding = VectorField(dimensions=768, null=True)

    class Meta:
        abstract = True

    @classmethod
    def search(cls, query, max_distance: float | None = None, top_k: int |None = None):
        max_distance = max_distance or VECTOR_SETTINGS["max_distance"]
        top_k = top_k or VECTOR_SETTINGS.get("top_k", None)
        vector = get_embedding_model().encode(query) if isinstance(query, str) else query
        distance = CosineDistance("embedding", vector)
        res = (
            cls.objects.annotate(distance=distance)
            .filter(distance__lte=max_distance)
            .order_by("distance")
        )
        return res[:top_k] if top_k else res