from abc import ABC

from django.db.models import QuerySet

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
        return model.search(skill_description)

    def extract_skills(self, user_inputs: str | list[str]) -> list:
        """Extract a list of skills from a user input or list of inputs."""
        raise NotImplementedError

    def extract_occupation(self, user_input: str):
        """Extract the occupation from a given user input."""
        return self.match_with_kb(self.occupation_kb, user_input)
