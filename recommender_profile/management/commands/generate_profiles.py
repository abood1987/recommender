from django.core.management.base import BaseCommand
from recommender_profile.models import UserProfile, TaskProfile


class Command(BaseCommand):
    help = "Generate profiles embeddings"

    def add_arguments(self, parser):
        parser.add_argument("--ignore-occ", action="store_true", dest="ignore_occupations", help="ignore the import of the occupations.")
        parser.add_argument("--ignore-skills", action="store_true", dest="ignore_skills", help="ignore the import of the skills.")

    def handle(self, *args, **options):
        self.stdout.write("---START---")

        if not options.get("ignore_occupations", False):
            self.stdout.write("---Start generating of embeddings of Tasks----")
            self.generate_task_embeddings()
            self.stdout.write("---Embeddings of Tasks generated----")

        if not options.get("ignore_skills", False):
            self.stdout.write("---Start generating of embeddings of users----")
            self.generate_user_embeddings()
            self.stdout.write("---Embeddings of users generated----")
        self.stdout.write("---END---")

    def generate_user_embeddings(self):
        i = 0
        for q in UserProfile.objects.all():
            if i % 1000 == 0:
                print(i)
            q.kb_matching_and_generate_embedding()
            i += 1

    def generate_task_embeddings(self):
        i = 0
        for q in TaskProfile.objects.all():
            if i % 1000 == 0:
                print(i)
            q.kb_matching_and_generate_embedding()
            i += 1