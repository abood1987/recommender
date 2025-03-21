import torch
import pandas as pd
import numpy as np
from django.core.management.base import BaseCommand
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from recommender_kb.models import Skill


class Command(BaseCommand):
    help = "Export ESCO skills & occupations with embeddings"
    requires_system_checks = []

    def handle(self, *args, **options):
        self.stdout.write("---START---")
        self.sentence_transformer = SentenceTransformer('paraphrase-mpnet-base-v2')
        self.jobbert_base_tokenizer = AutoTokenizer.from_pretrained("jjzha/jobbert-base-cased")
        self.jobbert_base_model = AutoModel.from_pretrained("jjzha/jobbert-base-cased")

        df = pd.DataFrame(list(Skill.objects.values('id', 'broader_id', 'description', 'alt_labels', 'label', 'embedding')))
        df['combined'] = df.apply(lambda row: f"{row['description']}, {row['label']}, " + ", ".join(row['alt_labels']), axis=1)
        df['combined_label'] = df.apply(lambda row: f"{row['label']}, " + ", ".join(row['alt_labels']), axis=1)

        df["embedding"] = df["combined"].apply(self.get_sem_embeddings)
        df["embedding_labels"] = df["combined_label"].apply(self.get_sem_embeddings)
        df["jobbert"] = df["combined"].apply(self.get_jobbert_embedding)
        df["jobbert_labels"] = df["combined_label"].apply(self.get_jobbert_embedding)

        np.savez_compressed(
            'esco_sem_jobbert.npz',
            label=np.array(df['label']),
            embedding=np.array(df['embedding'].tolist()),
            jobbert=np.array(df['jobbert'].tolist()),
            embedding_labels=np.array(df['embedding_labels'].tolist()),
            jobbert_labels=np.array(df['jobbert_labels'].tolist())
        )
        self.stdout.write("---END---")

    def get_sem_embeddings(self, x):
        return self.sentence_transformer.encode(x, normalize_embeddings=True)

    def get_jobbert_embedding(self, text):
        inputs = self.jobbert_base_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.jobbert_base_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding