import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from recommender_core.matcher.base import MatcherBase


class TFIDFMatcher(MatcherBase):
    def _get_recommendations(self, users_df: pd.DataFrame, tasks_df: pd.DataFrame) -> dict:
        def list_to_text(lst):
            return " ".join(lst)

        corpus = pd.concat([users_df["standard_skills"], tasks_df["standard_skills"]]).apply(list_to_text)
        vectorizer = TfidfVectorizer()
        vectorizer.fit(corpus)
        user_tfidf = vectorizer.transform(users_df["standard_skills"].apply(list_to_text))
        job_tfidf = vectorizer.transform(tasks_df["standard_skills"].apply(list_to_text))
        sim_matrix_tfidf = cosine_similarity(user_tfidf, job_tfidf)

        task_ids = tasks_df["id"].tolist()

        def get_matches(row):
            indices = np.where(row >= self.threshold)[0]
            sorted_indices = indices[np.argsort(row[indices])[::-1][:self.top_k]]
            return [task_ids[i] for i in sorted_indices]

        users_df["matched_jobs"] = [get_matches(row) for row in sim_matrix_tfidf]
        #
        # users_df['matched_jobs'] = [
        #     list(tasks_df.iloc[np.where(row >= self.threshold)[0], "id"])
        #     for row in sim_matrix_tfidf
        # ]
        return users_df.set_index("id")["matched_jobs"].to_dict()