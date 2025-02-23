from abc import ABC
from recommender_core.utils.singleton import Singleton


class BaseExtractorModel(Singleton, ABC):


    def __init__(self, model_name: str, model_path: str | None = None):
        self.model_name = model_name
        self.model_path = model_path

    def start_prompt(self, prompt):
        """
        Start direct prompt using selected model.
        """
        raise NotImplementedError

    def match_with_kb(self, model, skill_description):
        """
        Match instance description with existing instances in the knowledge base using cosine similarity.
        """
        raise NotImplementedError

    def get_standard_skills(self, user_input: str) -> list:
        """
        Complete process: Extract skills from user input and Generate description and match with the knowledge base.
        """
        raise NotImplementedError

    def get_standard_occupation(self, user_input: str):
        """
        Complete process: Generate occupation description and match with the knowledge base.
        """
        raise NotImplementedError

    def encode(self, text) -> list[float]:
        raise NotImplementedError