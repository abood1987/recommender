from itertools import chain
from typing import Union, List
import warnings
import pickle
import os
import re

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

from recommender_core.embeddings.base import BaseEmbeddingModel
from recommender_core.utils.helper import get_embedding_model
from recommender_kb.models import Skill, Occupation


class KBExtractor:
    def __init__(self, threshold: float = 0.45):

        self.threshold = threshold

        self._model: BaseEmbeddingModel = get_embedding_model()
        self._load_skills()
        self._load_occupations()
        self._create_skill_embeddings()
        self._create_occupation_embeddings()

    def _load_skills(self):
        self._skills = pd.DataFrame(list(Skill.objects.values("label", "embedding")))
        self._skills.rename(columns={"label": "id", "description": "embedding"}, inplace=True)
        self._skill_ids = self._skills["id"].to_numpy()

    def _load_occupations(self):
        self._occupations = pd.DataFrame(list(Occupation.objects.values("label", "embedding")))
        self._occupations.rename(columns={"label": "id", "description": "embedding"}, inplace=True)
        self._occupation_ids = self._occupations["id"].to_numpy()

    def _create_skill_embeddings(self):
        self._skill_embeddings = np.array(self._skills["embedding"].to_list())

    def _create_occupation_embeddings(self):
        self._occupation_embeddings = np.array(self._occupations["embedding"].to_list())

    @staticmethod
    def text_to_sentences(text: str) -> List[str]:
        return [s for s in re.split(r"\r|\n|\t|\.|\,|\;|and|or", text.strip()) if s]

    def _get_entity(
        self,
        texts: List[str],
        entity_ids: np.ndarray[str],
        entity_embeddings: torch.Tensor | np.ndarray,
    ) -> List[List[str]]:

        if all(not text for text in texts):
            return [[] for _ in texts]

        # Split the texts into sentences and then flatten them to perform calculations faster
        texts = [self.text_to_sentences(text) for text in texts]
        sentences = list(chain.from_iterable(texts))

        # Calculate the embeddings for all flattened sentences
        sentence_embeddings = self._model.encode(sentences)

        # Calculate the similarity between all flattened sentences and all entities and
        # find the most similar entity for each sentence.
        # The embeddings are normalized so the dot product is the cosine similarity
        similarity_matrix = util.dot_score(sentence_embeddings, entity_embeddings)
        most_similar_entity_scores, most_similar_entity_indices = torch.max(
            similarity_matrix, dim=-1
        )

        # Un-flatten the list of most similar entities to match the original texts
        entity_ids_per_text = []
        sentences = 0

        for text in texts:
            sentences_in_text = len(text)

            most_similar_entity_indices_text = most_similar_entity_indices[
                sentences : sentences + sentences_in_text
            ]
            most_similar_entity_scores_text = most_similar_entity_scores[
                sentences : sentences + sentences_in_text
            ]

            # Filter the entities based on the threshold
            most_similar_entity_indices_text = (
                most_similar_entity_indices_text[
                    torch.nonzero(most_similar_entity_scores_text > self.threshold)
                ]
                .squeeze(dim=-1)
                .unique()
                .tolist()
            )

            # Create a list of dictionaries containing the entities for the current text
            entity_ids_per_text.append(
                np.take(entity_ids, most_similar_entity_indices_text).tolist()
            )

            sentences += sentences_in_text

        return entity_ids_per_text

    def get_skills(self, texts: List[str]) -> List[List[str]]:

        return self._get_entity(
            texts,
            self._skill_ids,
            self._skill_embeddings,
        )

    def get_occupations(self, texts: List[str]) -> List[List[str]]:

        return self._get_entity(
            texts,
            self._occupation_ids,
            self._occupation_embeddings,
        )