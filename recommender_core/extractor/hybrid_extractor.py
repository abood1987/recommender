from recommender_core.extractor.base import ExtractorBase
from recommender_core.utils.collector import ClassTracer


class HybridExtractor(ExtractorBase):
    @ClassTracer.exclude
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.extract_prompt= kwargs["extract_prompt"]
        self.description_prompt = kwargs["description_prompt"]
        self.occupation_extract_prompt = kwargs["occupation_extract_prompt"]

    def extract_skills(self, user_inputs: str | list[str]) -> list:
        user_inputs = user_inputs if isinstance(user_inputs, list) else [user_inputs]
        standard_skills = []

        for user_input in user_inputs:
            matches_skills = list(self.match_with_kb(self.skill_kb, user_input))
            if not matches_skills:
                matches_skills = self._llm_fallback(user_input)
            standard_skills.extend(matches_skills)
        return list(filter(None, list(set(standard_skills))))

    def _llm_fallback(self, user_input: str) -> list:
        standard_skills = []
        skills_str = self.llm.prompt(self.extract_prompt % user_input)
        for skill in skills_str.split(','):
            description = self.llm.prompt(self.description_prompt % skill)
            standard_skills.extend(list(self.match_with_kb(self.skill_kb, description)))
        return standard_skills

    def extract_occupation(self, user_input: str):
        matches_occupation = super().extract_occupation(user_input)
        if not matches_occupation.exists():
            return self.match_with_kb(
                self.occupation_kb,
                self.llm.prompt(self.occupation_extract_prompt % user_input)
            )
        return matches_occupation

