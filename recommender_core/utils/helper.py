from django.db.models import QuerySet
from django.utils.module_loading import import_string
from pgvector.django import CosineDistance
from recommender.settings import VECTOR_SETTINGS
from recommender_core.embeddings.base import BaseEmbeddingModel
from recommender_core.matcher.base import BaseMatcherModel


def get_embedding_model() -> BaseEmbeddingModel:
    """
    Get the embedding model, as specified in the settings.
    :return: embedding model instance.
    """
    model_cls = VECTOR_SETTINGS["embeddings"]["class"]
    if isinstance(model_cls, str):
        model_cls = import_string(model_cls)
    model_config = VECTOR_SETTINGS["embeddings"]["configuration"]
    return model_cls(**model_config)


def get_llm_model() -> BaseMatcherModel:
    """
    Get matcher model, as specified in the settings.
    :return: matcher model instance.
    """
    model_cls = VECTOR_SETTINGS["matcher"]["class"]
    if isinstance(model_cls, str):
        model_cls = import_string(model_cls)
    model_config = VECTOR_SETTINGS["matcher"]["configuration"]
    return model_cls(**{
        **model_config,
        "embedding_model": get_embedding_model()
    })


def model_semantic_search(model, query: str | list[float], max_distance: float | None = None) -> QuerySet:
    max_distance = max_distance or VECTOR_SETTINGS["max_distance"]
    vector = get_embedding_model().encode(query) if isinstance(query, str) else query
    distance = CosineDistance("embedding", vector)
    return (
        model.objects.annotate(distance=distance)
        .filter(distance__lte=max_distance)
        .order_by("distance")
    )