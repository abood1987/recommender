import re
from recommender_core.extractor.base import ExtractorBase


class EmbeddingSimilarityExtractor (ExtractorBase):
    def extract_skills(self, user_inputs: str | list[str]) -> list:
        user_inputs = user_inputs if isinstance(user_inputs, list) else [user_inputs]
        standard_skills = []

        for user_input in user_inputs:
            standard_skills.extend(list(self.match_with_kb(self.skill_kb, user_input)))
        return list(filter(None, standard_skills))
