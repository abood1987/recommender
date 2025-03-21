from abc import ABC

from django.apps import apps

from recommender_core.embeddings.base import EmbeddingModelBase
from recommender_core.models import BaseVectorModel
from recommender_core.utils.singleton import Singleton


class MatcherBase(Singleton, ABC):
    def __init__(self, embedding_model: EmbeddingModelBase, llm_model, **kwargs):
        self.embedding_model = embedding_model
        self.llm = llm_model
        self.skill_kb: "BaseVectorModel" = apps.get_model(app_label="recommender_kb", model_name="Skill")
        self.occupation_kb: "BaseVectorModel" = apps.get_model(app_label="recommender_kb", model_name="Occupation")

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