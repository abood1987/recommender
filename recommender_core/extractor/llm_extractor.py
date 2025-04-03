from recommender_core.extractor.base import ExtractorBase
from recommender_core.utils.collector import ClassTracer


class LLMExtractor(ExtractorBase):
    @ClassTracer.exclude
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.extract_prompt= kwargs.get("extract_prompt", "Extract professional skills from the following text: %s. Return only skills, comma-separated.")
        self.description_prompt = kwargs.get("description_prompt", "Generate a detailed professional description for the skill: %s.")

    def extract_skills(self, user_inputs: str | list[str]) -> list:
        user_inputs = user_inputs if isinstance(user_inputs, list) else [user_inputs]
        standard_skills = []

        for user_input in user_inputs:
            skills_str = self.llm.prompt(self.extract_prompt % user_input)
            for skill in skills_str.split(','):
                description = self.llm.prompt(self.description_prompt % skill)
                standard_skills.append(self.match_with_kb(self.skill_kb, description))
        return list(filter(None, standard_skills))
