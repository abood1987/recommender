from pathlib import Path

from transformers import T5Tokenizer, T5ForConditionalGeneration
from recommender_core.embeddings.base import BaseEmbeddingModel
from recommender_core.extractor.base import BaseExtractorModel
from django.apps import apps


class FlanT5Model(BaseExtractorModel):
    def __init__(
            self,
            model_name: str,
            skill_description_prompt: str,
            extract_skills_prompt: str,
            occupation_description_prompt: str,
            embedding_model: BaseEmbeddingModel,
            model_path: str | None = None
    ):
        # Initialize FLAN-T5 model and tokenizer
        # self.model_name = "google/flan-t5-large"
        self.model_path = model_path
        self.model_name = model_name
        self.skill_description_prompt = skill_description_prompt
        self.extract_skills_prompt = extract_skills_prompt
        self.occupation_description_prompt = occupation_description_prompt
        self.tokenizer, self.model = self._get_model()
        self.embedding_model = embedding_model

    def _get_model(self):
        if not self.model_path:
            return (
                T5Tokenizer.from_pretrained(self.model_name),
                T5ForConditionalGeneration.from_pretrained(self.model_name)
            )

        path = Path(self.model_path)
        if path.exists() and any(path.iterdir()):
            return (
                T5Tokenizer.from_pretrained(self.model_path),
                T5ForConditionalGeneration.from_pretrained(self.model_path)
            )

    def start_prompt(self, prompt):
        """
        Start direct prompt using FLAN-T5.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(inputs.input_ids, max_length=512, num_beams=5, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


    def match_with_kb(self, model, skill_description: str):
        matched_objects = model.search(skill_description)
        return matched_objects.first() if matched_objects.exists() else None

    def _get_standard_skills(self, user_input: str) -> list:
        # Step 1: Extract skills from user input
        skills_str = self.start_prompt(self.extract_skills_prompt % user_input)
        skills_list = skills_str.split(',')

        model = apps.get_model(app_label="recommender_kb", model_name="Skill")
        matched_skills = []
        for skill in skills_list:
            description = self.start_prompt(self.skill_description_prompt % skill)
            matched_skills.append(self.match_with_kb(model, description))
        return matched_skills

    def get_standard_skills(self, user_input: str | list[str]) -> list:
        standard_skills = []
        if isinstance(user_input, str):
            standard_skills = self._get_standard_skills(user_input)
        elif isinstance(user_input, list):
            for skill in user_input:
                standard_skills.extend(self._get_standard_skills(skill))
        return standard_skills

    def get_standard_occupation(self, user_input: str):
        model = apps.get_model(app_label="recommender_kb", model_name="Occupation")
        description = self.start_prompt(self.occupation_description_prompt % user_input)
        return self.match_with_kb(model, description)