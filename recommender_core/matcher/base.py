from abc import ABC
from functools import lru_cache

import pandas as pd
from django.apps import apps
from django.db.models import QuerySet

from recommender.settings import VECTOR_SETTINGS
from recommender_core.embeddings.base import EmbeddingModelBase
from recommender_core.models import BaseVectorModel
from recommender_core.utils.singleton import Singleton
from recommender_profile.models import UserProfile, TaskProfile


class MatcherBase(Singleton, ABC):
    def __init__(self, embedding_model: EmbeddingModelBase, llm_model, **kwargs):
        self.kwargs = kwargs
        self.embedding_model = embedding_model
        self.llm = llm_model
        self.include_broader = kwargs.get('include_broader', False)
        self.threshold = 1 - VECTOR_SETTINGS["max_distance"]
        self.skill_kb: "BaseVectorModel" = apps.get_model(app_label="recommender_kb", model_name="Skill")
        self.occupation_kb: "BaseVectorModel" = apps.get_model(app_label="recommender_kb", model_name="Occupation")

    @lru_cache(maxsize=None)
    def _users_df(self):
        df = pd.DataFrame.from_records(
            UserProfile.objects.values(
                "id",
                "standard_skills__label",
                "standard_skills__broader__label"
            )
        )

        if self.include_broader:
            # Set of skills that are themselves used as a broader
            broader_set = set(df["standard_skills__broader__label"].dropna())

            # Expand only if the skill is NOT a broader of others
            def expand(row):
                skill = row["standard_skills__label"]
                broader = row["standard_skills__broader__label"]
                if skill in broader_set:
                    return [skill]
                return [skill, broader] if broader else [skill]

            df["skills"] = df.apply(expand, axis=1)
        else:
            # No expansion: keep only the original skill
            df["skills"] = df["standard_skills__label"].apply(lambda x: [x])

        # Group by user and flatten lists
        group_df = (
            df.groupby("id")["skills"]
            .apply(lambda x: sum(x, []))  # flatten list of lists
            .reset_index()
            .rename(columns={"skills": "standard_skills"})
        )
        return group_df

    @lru_cache(maxsize=None)
    def _tasks_df(self):
        # Step 1: Query all Tasks and related skills and fallback skills
        df = pd.DataFrame.from_records(
            TaskProfile.objects.values(
                "id",
                "standard_skills__label",
                "standard_skills__broader__label",
                "standard_title__skills__label",
                "standard_title__skills__broader__label"
            )
        )

        if self.include_broader:
            # Set of skills that are used as a broader for other skills
            broader_set = set(
                df["standard_skills__broader__label"].dropna().tolist()
                + df["standard_title__skills__broader__label"].dropna().tolist()
            )

            def get_skills(row):
                skill = row["standard_skills__label"]
                broader = row["standard_skills__broader__label"]
                fallback = row["standard_title__skills__label"]
                fallback_broader = row["standard_title__skills__broader__label"]

                if skill:  # primary skill
                    if skill in broader_set:
                        return [skill]
                    return [skill, broader] if broader else [skill]
                elif fallback:  # fallback skill
                    if fallback in broader_set:
                        return [fallback]
                    return [fallback, fallback_broader] if fallback_broader else [fallback]
                return []

        else:
            def get_skills(row):
                skills = row["standard_skills__label"]
                fallback_skills = row["standard_title__skills__label"]
                return (
                    [skills] if skills
                    else [fallback_skills] if fallback_skills
                    else []
                )

        df["skills"] = df.apply(get_skills, axis=1)

        group_df = (
            df.groupby("id")["skills"]
            .apply(lambda x: sum(x, []))  # flatten list of lists
            .reset_index()
            .rename(columns={"skills": "standard_skills"})
        )
        return group_df

    def _filter_df(self, df: pd.DataFrame, objects: QuerySet[BaseVectorModel] | BaseVectorModel | None = None) -> pd.DataFrame:
        if objects is None:
            return df
        ids = (
            [objects.id] if isinstance(objects, BaseVectorModel)
            else list(objects.values_list("id", flat=True))
        )
        return df[df["id"].isin(ids)]

    def get_recommendations(
            self,
            users: QuerySet[UserProfile] | UserProfile | None = None,
            tasks: QuerySet[TaskProfile] | TaskProfile | None = None,
    ) -> dict:
        users_df = self._filter_df(self._users_df(), users)
        tasks_df = self._filter_df(self._tasks_df(), tasks)

        if users_df.empty or tasks_df.empty:
            return {}
        return self._get_recommendations(users_df, tasks_df)

    def _get_recommendations(self, users_df: pd.DataFrame, tasks_df: pd.DataFrame) -> dict:
        raise NotImplementedError
