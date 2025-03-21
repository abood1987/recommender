from abc import ABC
from recommender_core.embeddings.base import EmbeddingModelBase
from recommender_core.utils.collector import DataCollector, ClassTracer
from recommender_core.utils.singleton import Singleton
from typing import TYPE_CHECKING
from django.apps import apps
if TYPE_CHECKING: # solve circular import
    from recommender_core.models import BaseVectorModel



class ExtractorBase(Singleton, ABC, metaclass=ClassTracer):
    @ClassTracer.exclude
    def __init__(self, **kwargs):
        self.embedding_model: EmbeddingModelBase = kwargs["embedding_model"]
        self.llm = kwargs["llm_model"]
        self.skill_kb: "BaseVectorModel" = apps.get_model(app_label="recommender_kb", model_name="Skill")
        self.occupation_kb: "BaseVectorModel" = apps.get_model(app_label="recommender_kb", model_name="Occupation")
        self.collector = DataCollector()

    @staticmethod
    def match_with_kb(model: "BaseVectorModel", skill_description: str):
        """
        Match instance description with existing instances in the knowledge base using cosine similarity.
        """
        matched_objects = model.search(skill_description)
        return matched_objects.first() if matched_objects.exists() else None

    def extract_skills(self, user_inputs: str | list[str]) -> list:
        raise NotImplementedError

    def extract_occupation(self, user_input: str):
        raise NotImplementedError
