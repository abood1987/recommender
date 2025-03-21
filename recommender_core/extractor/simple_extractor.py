import re
from recommender_core.extractor.base import ExtractorBase


class SimpleExtraction(ExtractorBase):
    def split_text_statements(self, text: str) -> list[str]:
        """Splits a text into smaller statements."""
        if not isinstance(text, str) or not text.strip(): return []  # Return empty list for invalid inputs
        return [s.strip() for s in re.split(r"[\r\n\t.,;]| and | or ", text) if s] + [text]

    def extract_skills(self, user_inputs: str | list[str]) -> list:
        user_inputs = user_inputs if isinstance(user_inputs, list) else [user_inputs]
        standard_skills = []

        for user_input in user_inputs:
            skills_list = self.split_text_statements(user_input)
            for skill in skills_list:
                standard_skills.append(self.match_with_kb(self.skill_kb, skill))
        return list(filter(None, standard_skills))

    def extract_occupation(self, user_input: str):
        return self.match_with_kb(self.occupation_kb, user_input)