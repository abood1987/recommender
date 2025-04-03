from functools import cache
import json
from django.utils.module_loading import import_string
from recommender.settings import VECTOR_SETTINGS
from recommender_core.embeddings.base import EmbeddingModelBase
from recommender_core.extractor.base import ExtractorBase
from recommender_core.llm.base import LLMModelBase
from recommender_core.matcher.base import MatcherBase


@cache
def get_embedding_model(settings=None) -> EmbeddingModelBase:
    """
    Get the embedding model, as specified in the settings.
    :return: embedding model instance.
    """
    settings = json.loads(settings) if settings else VECTOR_SETTINGS
    settings = settings.get("embeddings", VECTOR_SETTINGS["embeddings"])
    model_cls = settings["class"]
    if isinstance(model_cls, str):
        model_cls = import_string(model_cls)
    model_config = settings["configuration"]
    return model_cls(**model_config)


@cache
def get_llm_model(settings=None) -> LLMModelBase:
    """
    Get llm model, as specified in the settings.
    :return: llm model instance.
    """
    settings = json.loads(settings) if settings else VECTOR_SETTINGS
    settings = settings.get("llm", VECTOR_SETTINGS["llm"])
    model_cls = settings["class"]
    if isinstance(model_cls, str):
        model_cls = import_string(model_cls)
    model_config = settings["configuration"]
    return model_cls(**model_config)


@cache
def get_extractor(settings=None) -> ExtractorBase:
    """
    Get extractor model, as specified in the settings.
    :return: extractor model instance.
    """

    settings = json.loads(settings) if settings else VECTOR_SETTINGS
    extractor_settings = settings["extractor"]
    model_cls = extractor_settings["class"]
    if isinstance(model_cls, str):
        model_cls = import_string(model_cls)
    model_config = extractor_settings["configuration"]
    return model_cls(**{
        **model_config,
        "embedding_model": get_embedding_model(json.dumps(settings)),
        "llm_model": get_llm_model(json.dumps(settings)),
    })


@cache
def get_matcher(settings=None) -> MatcherBase:
    """
    Get matcher model, as specified in the settings.
    :return: matcher model instance.
    """

    settings = json.loads(settings) if settings else VECTOR_SETTINGS
    extractor_settings = settings["matcher"]
    model_cls = extractor_settings["class"]
    if isinstance(model_cls, str):
        model_cls = import_string(model_cls)
    model_config = extractor_settings["configuration"]
    return model_cls(**{
        **model_config,
        "embedding_model": get_embedding_model(json.dumps(settings)),
        "llm_model": get_llm_model(json.dumps(settings)),
    })