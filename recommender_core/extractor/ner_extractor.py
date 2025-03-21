from transformers import pipeline
from recommender_core.extractor.base import ExtractorBase
from recommender_core.utils.collector import ClassTracer


class NERExtractor(ExtractorBase):
    @ClassTracer.exclude
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.skills_model = pipeline(model=kwargs["skills_model"], aggregation_strategy="first")
        self.knowledge_model = pipeline(model=kwargs["knowledge_model"], aggregation_strategy="first")

    def extract_skills(self, user_inputs: str | list[str]) -> list:
        user_inputs = user_inputs if isinstance(user_inputs, list) else [user_inputs]
        standard_skills = []

        for user_input in user_inputs:
            for skill in self.ner(user_input):
                standard_skills.append(self.match_with_kb(self.skill_kb, skill))
        return list(filter(None, standard_skills))

    def extract_occupation(self, user_input: str):
        # return self.match_with_kb(self.occupation_kb, ", ".join(self.ner(user_input)))
        return self.match_with_kb(self.occupation_kb, user_input)

    def aggregate_span(self, results):
        new_results = []
        current_result = results[0]

        for result in results[1:]:
            if result["start"] == current_result["end"] + 1:
                current_result["word"] += " " + result["word"]
                current_result["end"] = result["end"]
            else:
                new_results.append(current_result)
                current_result = result

        new_results.append(current_result)
        return new_results

    def ner(self, text):
        output_skills = self.skills_model(text)
        for result in output_skills:
            if result.get("entity_group"):
                result["entity"] = "Skill"
                del result["entity_group"]

        output_knowledge = self.knowledge_model(text)
        for result in output_knowledge:
            if result.get("entity_group"):
                result["entity"] = "Knowledge"
                del result["entity_group"]

        if len(output_skills) > 0: output_skills = self.aggregate_span(output_skills)
        if len(output_knowledge) > 0: output_knowledge = self.aggregate_span(output_knowledge)
        res = [*output_skills, *output_knowledge]
        return [d["word"] for d in res]