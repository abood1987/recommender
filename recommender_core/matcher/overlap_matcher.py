import pandas as pd
from recommender_core.matcher.base import MatcherBase


class OverlapMatcher(MatcherBase):
    def overlap(self, u, j):
        return len(u & j) / len(j) if j else 0

    def _get_recommendations(self, users_df: pd.DataFrame, tasks_df: pd.DataFrame) -> dict:
        tasks_ids = tasks_df["id"].tolist()
        tasks_skills = tasks_df["standard_skills"].tolist()

        users_df["matched_jobs"] = users_df["standard_skills"].apply(
            lambda skills: [
                tasks_ids[i] for i, job_set in enumerate(tasks_skills)
                if job_set and self.overlap(set(skills), job_set) >= self.threshold
            ]
        )
        return users_df.set_index("id")["matched_jobs"].to_dict()