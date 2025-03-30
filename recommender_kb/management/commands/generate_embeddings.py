from django.core import checks
from django.core.management.base import BaseCommand
from recommender_core.embeddings.base import EmbeddingModelBase
from recommender_core.utils.helper import get_embedding_model
from recommender_kb.models import (
    Occupation,
    Skill, SkillGroup,
)


class Command(BaseCommand):
    help = "Generate ESCO embeddings"
    requires_system_checks = []

    def check(self, app_configs=None, tags=None, display_num_errors=False, include_deployment_checks=False,
              fail_level=checks.ERROR, databases=None):
        self.stdout.write(self.style.WARNING("'Generate ESCO embeddings': SKIPPING SYSTEM CHECKS!\n"))

    def __init__(self, stdout=None, stderr=None, no_color=False, force_color=False):
        super().__init__(stdout, stderr, no_color, force_color)
        self.embedding_model: EmbeddingModelBase = get_embedding_model()

    def add_arguments(self, parser):
        parser.add_argument("--clear", action="store_true", help="Delete old embeddings")

    def handle(self, *args, **options):
        self.stdout.write("---START---")

        if options.get("clear", False):
            self.clear_data()
            self.stdout.write("---Old data cleared----")

        self.generate_groups_embeddings()
        self.stdout.write("---Embeddings of Isco groups generated----")

        self.generate_occupations_embeddings()
        self.stdout.write("---Embeddings of occupations generated----")

        self.generate_skills_embeddings()
        self.stdout.write("---Embeddings of skills generated----")
        self.stdout.write("---END---")

    def clear_data(self):
        Occupation.objects.all().update(embedding=None)
        Skill.objects.all().update(embedding=None)
        SkillGroup.objects.all().update(embedding=None)

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

    def generate_groups_embeddings(self):
        embedding_fields = {"description": str, "label": str}
        self.update_model(SkillGroup, embedding_fields)

    def generate_occupations_embeddings(self):
        embedding_fields = {"description": str, "label": str, "alt_labels": list}
        self.update_model(Occupation, embedding_fields)

    def generate_skills_embeddings(self):
        embedding_fields = {"description": str, "label": str, "alt_labels": list}
        self.update_model(Skill, embedding_fields)
