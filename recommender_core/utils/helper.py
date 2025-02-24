from django.utils.module_loading import import_string
from recommender.settings import VECTOR_SETTINGS
from recommender_core.embeddings.base import BaseEmbeddingModel
from recommender_core.extractor.base import BaseExtractorModel


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


def get_llm_model() -> BaseExtractorModel:
    """
    Get extractor model, as specified in the settings.
    :return: extractor model instance.
    """
    model_cls = VECTOR_SETTINGS["extractor"]["class"]
    if isinstance(model_cls, str):
        model_cls = import_string(model_cls)
    model_config = VECTOR_SETTINGS["extractor"]["configuration"]
    return model_cls(**{
        **model_config,
        "embedding_model": get_embedding_model()
    })
