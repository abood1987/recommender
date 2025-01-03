from transformers import T5Tokenizer, T5ForConditionalGeneration
from recommender_core.embeddings.base import BaseEmbeddingModel
from recommender_core.matcher.base import BaseMatcherModel
from django.apps import apps
from openai import ChatCompletion


class T5MatcherModel(BaseMatcherModel):
    def __init__(
            self,
            model_name: str,
            skill_description_prompt: str,
            extract_skills_prompt: str,
            embedding_model: BaseEmbeddingModel
    ):
        # Initialize FLAN-T5 model and tokenizer
        # self.model_name = "google/flan-t5-large"
        self.skill_description_prompt = skill_description_prompt
        self.extract_skills_prompt = extract_skills_prompt
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.embedding_model = embedding_model

    def start_prompt(self, prompt):
        """
        Start direct prompt using FLAN-T5.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(inputs.input_ids, max_length=512, num_beams=5, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


    def match_skill_with_kb(self, skill_description):
        """
        Match the skill description with existing skills in the knowledge base using cosine similarity.
        """
        # Query database for the closest match
        skills_model = apps.get_model(app_label="recommender_kb", model_name="Skill")
        matched_skill = skills_model.search(skill_description)
        if matched_skill.exists():
            matched_skill = matched_skill.first()
            return matched_skill.label, matched_skill.distance
        return None, None

    def _get_standard_skills(self, user_input: str) -> list:
        # Step 1: Extract skills from user input
        skills_str = self.start_prompt(self.extract_skills_prompt % user_input)
        skills_list = skills_str.split(',')

        # Step 2: Generate description
        output_dict = {}
        matched_skills = []
        for skill in skills_list:
            description = self.start_prompt(self.skill_description_prompt % skill)
            # Step 2: Match with KB
            matched_skill, distance = self.match_skill_with_kb(description)
            matched_skills.append(matched_skill)
            # output_dict[skill] = {
            #     "description": description,
            #     "matched_skill": matched_skill,
            #     "distance": distance
            # }

        # Step 3: Return results
        return matched_skills

    def get_standard_skills(self, user_input: str | list[str]) -> list:
        standard_skills = []
        if isinstance(user_input, str):
            standard_skills = self._get_standard_skills(user_input)
        elif isinstance(user_input, list):
            for skill in user_input:
                standard_skills.extend(self._get_standard_skills(skill))
        return standard_skills