from django.db import models
from pgvector.django import VectorField

from recommender_core.utils.helper import model_semantic_search


class BaseVectorModel(models.Model):
    embedding = VectorField(dimensions=768, null=True)

    class Meta:
        abstract = True

    @classmethod
    def search(cls, query, max_distance: float | None = None):
        return model_semantic_search(cls, query, max_distance)