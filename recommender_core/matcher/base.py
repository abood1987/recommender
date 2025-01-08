from abc import ABC
from recommender_core.utils.singleton import Singleton


class BaseMatcherModel(Singleton, ABC):

    def match_by_text(self, model, skill_description):
        """
        Match instance description with existing instances in the knowledge base using cosine similarity.
        """
        raise NotImplementedError

    def match_by_embedding(self, model, skill_description):
        """
        Match instance description with existing instances in the knowledge base using cosine similarity.
        """
        raise NotImplementedError