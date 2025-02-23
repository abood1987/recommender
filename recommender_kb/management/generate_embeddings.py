from django.core.management.base import BaseCommand
from recommender_core.embeddings.base import BaseEmbeddingModel
from recommender_core.utils.helper import get_embedding_model
from recommender_kb.models import (
    Occupation,
    Skill, ISCOGroup,
)


class Command(BaseCommand):
    help = "Generate ESCO embeddings"

    def __init__(self, stdout=None, stderr=None, no_color=False, force_color=False):
        super().__init__(stdout, stderr, no_color, force_color)
        self.embedding_model: BaseEmbeddingModel = get_embedding_model()

    def add_arguments(self, parser):
        parser.add_argument("--clear", action="store_true", help="Delete old embeddings")
        parser.add_argument("--ignore-occ", action="store_true", dest="ignore_occupations", help="ignore the import of the occupations.")
        parser.add_argument("--ignore-skills", action="store_true", dest="ignore_skills", help="ignore the import of the skills.")
        parser.add_argument("--ignore-isco-groups", action="store_true", dest="ignore_isco_groups", help="ignore the import of the isco groups.")

    def handle(self, *args, **options):
        self.stdout.write("---START---")

        if options.get("clear", False):
            self.clear_data()
            self.stdout.write("---Old data cleared----")

        if not options.get("ignore_isco_groups", False):
            self.generate_isco_groups_embeddings()
            self.stdout.write("---Embeddings of Isco groups generated----")

        if not options.get("ignore_occupations", False):
            self.generate_occupations_embeddings()
            self.stdout.write("---Embeddings of occupations generated----")

        if not options.get("ignore_skills", False):
            self.generate_skills_embeddings()
            self.stdout.write("---Embeddings of skills generated----")
        self.stdout.write("---END---")

    def clear_data(self):
        Occupation.objects.all().update(embedding=None)
        Skill.objects.all().update(embedding=None)
        ISCOGroup.objects.all().update(embedding=None)

    def get_text(self, instance, fields_dict):
        text_list = []
        for f_name, f_type in fields_dict.items():
            if f_type == str:
                text_list.append(getattr(instance, f_name))
            elif f_type == list:
                text_list.extend(getattr(instance, f_name))
        return ", ".join(text_list)

    def update_model(self, model, fields):
        queryset = model.objects.all()
        for q in queryset:
            q.embedding = self.embedding_model.encode(self.get_text(q, fields))
        model.objects.bulk_update(queryset, ["embedding"])

    def generate_isco_groups_embeddings(self):
        # embedding_fields = {"label": str, "description": str}
        embedding_fields = {"description": str}
        self.update_model(ISCOGroup, embedding_fields)

    def generate_occupations_embeddings(self):
        # embedding_fields = {"label": str, "alt_labels": list, "hidden_labels": list, "description": str}
        embedding_fields = {"description": str}
        self.update_model(Occupation, embedding_fields)

    def generate_skills_embeddings(self):
        # embedding_fields = {"label": str, "alt_labels": list, "hidden_labels": list, "description": str}
        embedding_fields = {"description": str}
        self.update_model(Skill, embedding_fields)
