from functools import cached_property, lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from recommender_core.matcher.base import MatcherBase


class BinaryVectorMatcher(MatcherBase):
    @cached_property
    def _all_skills(self):
        return sorted(set(list(self.skill_kb.objects.values_list('label', flat=True))))

    @cached_property
    def _skill_to_idx(self):
        return {skill: i for i, skill in enumerate(self._all_skills)}

    def encode_binary(self, skills: list[str]):
        skills = sorted(set(skills))
        skill_to_idx = self._skill_to_idx
        vec = np.zeros(len(skill_to_idx))
        for skill in skills:
            if skill in skill_to_idx:
                vec[skill_to_idx[skill]] = 1
        return vec

    @lru_cache(maxsize=None)
    def _users_df(self):
        df = super()._users_df()
        df["vector"] = df["standard_skills"].apply(self.encode_binary)
        return df

    @lru_cache(maxsize=None)
    def _tasks_df(self):
        df = super()._tasks_df()
        df["vector"] = df["standard_skills"].apply(self.encode_binary)
        return df

    def _get_recommendations(self, users_df: pd.DataFrame, tasks_df: pd.DataFrame) -> dict:
        sim_matrix_cosine = cosine_similarity(np.stack(users_df["vector"].values), np.stack(tasks_df["vector"].values))
        task_ids = tasks_df["id"].tolist()

        def get_matches(row):
            indices = np.where(row >= self.threshold)[0]
            sorted_indices = indices[np.argsort(row[indices])[::-1][:self.top_k]]
            return [task_ids[i] for i in sorted_indices]

        users_df['matched_jobs'] = [get_matches(row) for row in sim_matrix_cosine]
        # users_df['matched_jobs'] = [
        #     tasks_df.iloc[np.where(row >= self.threshold)[0]]["id"].tolist()
        #     for row in sim_matrix_cosine
        # ]
        return users_df.set_index("id")["matched_jobs"].to_dict()