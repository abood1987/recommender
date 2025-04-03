import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz, utils
from recommender_core.matcher.base import MatcherBase


class FuzzyMatcher(MatcherBase):
    def fuzzy_match_user_to_job(self, user_skills, job_skills):
        if not user_skills or not job_skills:
            return 0.0

        # noinspection PyTypeChecker
        # Compute fuzzy similarity matrix
        sim_matrix = process.cdist(
            user_skills,
            job_skills,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=float(self.kwargs.get("fuzzy_threshold", 60)),
            dtype=np.uint8,
            processor=utils.default_process
        )

        matched_mask = np.any(sim_matrix, axis=1)
        return matched_mask.sum() / len(user_skills)

    def _get_recommendations(self, users_df: pd.DataFrame, tasks_df: pd.DataFrame) -> dict:
        tasks_ids = tasks_df["id"].tolist()
        tasks_skills = tasks_df["standard_skills"].tolist()

        def get_matches(user_skills):
            scores = [
                (tasks_ids[i], self.fuzzy_match_user_to_job(user_skills, job_sk))
                for i, job_sk in enumerate(tasks_skills)
            ]
            filtered = [(tid, score) for tid, score in scores if score >= self.threshold]
            sorted_filtered = sorted(filtered, key=lambda x: x[1], reverse=True)
            return [tid for tid, _ in sorted_filtered[:self.top_k]]

        users_df["matched_jobs"] = users_df["standard_skills"].apply(get_matches)
        # users_df["matched_jobs"] = users_df["standard_skills"].apply(
        #     lambda skills: [
        #         tasks_ids[i]
        #         for i, job_sk in enumerate(tasks_skills)
        #         if self.fuzzy_match_user_to_job(skills, job_sk) >= self.threshold
        #     ]
        # )
        return users_df.set_index("id")["matched_jobs"].to_dict()