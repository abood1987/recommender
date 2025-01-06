from abc import ABC
from recommender_core.utils.singleton import Singleton


class BaseMatcherModel(Singleton, ABC):

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

    def get_standard_skills(self, user_input: str):
        """
        Complete process: Extract skills from user input and Generate description and match with the knowledge base.
        """
        raise NotImplementedError

    def get_standard_occupation(self, user_input: str):
        """
        Complete process: Generate occupation description and match with the knowledge base.
        """
        raise NotImplementedError
