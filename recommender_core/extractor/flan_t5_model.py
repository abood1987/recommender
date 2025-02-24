from pathlib import Path

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from recommender_core.embeddings.base import BaseEmbeddingModel
from recommender_core.extractor.base import BaseExtractorModel
from django.apps import apps
from recommender_core.utils.collector import DataCollector, ClassTracer
from typing import TYPE_CHECKING

if TYPE_CHECKING: # solve circular import
    from recommender_core.models import BaseVectorModel


class FlanT5Model(BaseExtractorModel, metaclass=ClassTracer):
    @ClassTracer.exclude
    def __init__(
            self,
            model_name: str,
            skill_description_prompt: str,
            extract_skills_prompt: str,
            occupation_description_prompt: str,
            embedding_model: BaseEmbeddingModel,
            model_path: str | None = None
    ):
        super().__init__(model_name, model_path)
        # Initialize FLAN-T5 model and tokenizer
        # self.model_name = "google/flan-t5-large"
        self.skill_description_prompt = skill_description_prompt
        self.extract_skills_prompt = extract_skills_prompt
        self.occupation_description_prompt = occupation_description_prompt
        self.tokenizer, self.model = self._get_model()
        self.embedding_model = embedding_model
        self.skill_kb: "BaseVectorModel" = apps.get_model(app_label="recommender_kb", model_name="Skill")
        self.occupation_kb: "BaseVectorModel" = apps.get_model(app_label="recommender_kb", model_name="Occupation")
        self.collector = DataCollector()

    @ClassTracer.exclude
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
        with torch.no_grad(): # No training, just predictions => Disables gradient tracking (Faster & Memory Efficient)
            outputs = self.model.generate(inputs.input_ids, max_length=512, num_beams=5, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def match_with_kb(self, model: "BaseVectorModel", skill_description: str):
        matched_objects = model.search(skill_description)
        return matched_objects.first() if matched_objects.exists() else None

    def get_standard_skills(self, user_inputs: list[str]) -> list:
        standard_skills = []

        for user_input in user_inputs:
            skills_str = self.start_prompt(self.extract_skills_prompt % user_input)
            for skill in skills_str.split(','):
                description = self.start_prompt(self.skill_description_prompt % skill)
                standard_skills.append(self.match_with_kb(self.skill_kb, description))
        return list(filter(None, standard_skills))

    def get_standard_occupation(self, user_input: str):
        description = self.start_prompt(self.occupation_description_prompt % user_input)
        return self.match_with_kb(self.occupation_kb, description)

    @ClassTracer.exclude
    def encode(self, text) -> list[float]:
        return self.embedding_model.encode(text)