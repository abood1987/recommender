import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from recommender_core.matcher.base import MatcherBase


class EmbeddingMatcher(MatcherBase):

    def _get_recommendations(self, users_df: pd.DataFrame, tasks_df: pd.DataFrame) -> dict:
        # save the embedding in DB to speed up the process
        u_esco_emb = users_df["standard_skills"].apply(lambda x: self.embedding_model.encode(", ".join(x)))
        j_esco_emb = tasks_df["standard_skills"].apply(lambda x: self.embedding_model.encode(", ".join(x)))

        sim_matrix_emb = cosine_similarity(np.stack(u_esco_emb), np.stack(j_esco_emb))

        # Match jobs above threshold
        embedding_df = users_df.copy()
        embedding_df["matched_jobs"] = [
            list(tasks_df.loc[np.where(row >= self.threshold)[0], "id"])
            for row in sim_matrix_emb
        ]
        return users_df.set_index("id")["matched_jobs"].to_dict()