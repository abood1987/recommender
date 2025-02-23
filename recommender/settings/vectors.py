VECTOR_SETTINGS = {
    "max_distance": 0.45,
    "embeddings": {
        "class": "recommender_core.embeddings.SentenceTransformerModel",
        # Configuration is passed directly to the embeddings model class during initialization.
        "configuration": {
            "model_name": "paraphrase-mpnet-base-v2",
            "model_path": "C:\\Users\\abd19\\PycharmProjects\\recommender\\data\\models\\paraphrase-mpnet-base-v2"
        },
    },
    "extractor": {
        "class": "recommender_core.extractor.FlanT5Model",
        # Configuration is passed directly to the matcher model class during initialization.
        "configuration": {
            "model_name": "google/flan-t5-large",
            "model_path": "C:\\Users\\abd19\\PycharmProjects\\recommender\\data\\models\\google-flan-t5-large",
            "skill_description_prompt": "Generate a detailed professional description for the skill: %s.",
            "occupation_description_prompt": "Generate a detailed professional description for the occupation: %s.",
            "extract_skills_prompt": "Extract individual skills from the following text. Ensure each skill is clearly identified, professional, and in a normalized format:\n%s\nList of skills:"
        },
    },
}
