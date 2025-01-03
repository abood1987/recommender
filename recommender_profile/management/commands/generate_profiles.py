from django.core.management.base import BaseCommand
from recommender_core.embeddings.base import BaseEmbeddingModel
from recommender_core.utils.helper import get_embedding_model, get_llm_model
from recommender_profile.models import UserProfile, CompanyProfile

EMBEDDING_MODEL: BaseEmbeddingModel = get_embedding_model()


class Command(BaseCommand):
    help = "Generate profiles embeddings"

    def add_arguments(self, parser):
        parser.add_argument("--clear", action="store_true", help="Delete old embeddings")
        parser.add_argument("--ignore-occ", action="store_true", dest="ignore_occupations", help="ignore the import of the occupations.")
        parser.add_argument("--ignore-skills", action="store_true", dest="ignore_skills", help="ignore the import of the skills.")

    def handle(self, *args, **options):
        self.stdout.write("---START---")
        self.llm_model = get_llm_model()

        if options.get("clear", False):
            self.clear_data()
            self.stdout.write("---Old data cleared----")

        if not options.get("ignore_occupations", False):
            self.generate_occupations_embeddings()
            self.stdout.write("---Embeddings of occupations generated----")

        if not options.get("ignore_skills", False):
            self.generate_skills_embeddings()
            self.stdout.write("---Embeddings of skills generated----")
        self.stdout.write("---END---")

    def clear_data(self):
        CompanyProfile.objects.all().update(embedding=None)
        UserProfile.objects.all().update(embedding=None)

    def get_standard_skills(self, skills: list) -> list:
        text_list = []
        for skill in skills:
            res: dict = self.llm_model.process_skill(skills)
            if res["matched_skill"]:
                text_list.append(res["matched_skill"])
            else:
                print(f"standard skill not found: {skill}")
                # add original skill, so I can control them at the end
                text_list.append(f"-- {skill} --")
        return text_list

    def generate_occupations_embeddings(self):
        queryset = CompanyProfile.objects.all()
        i = 0
        for q in queryset:
            if i % 1000 == 0:
                print(i)
            standard_skills = self.get_standard_skills(q.skills)
            q.standard_skills = standard_skills
            q.embedding = EMBEDDING_MODEL.encode(", ".join(standard_skills))
            q.save()
            i += 1
        self.update_model(CompanyProfile, {"occupation": str})

    def generate_skills_embeddings(self):
        queryset = UserProfile.objects.all()
        i = 0
        for q in queryset:
            if i % 1000 == 0:
                print(i)
            standard_skills = self.get_standard_skills(q.skills)
            q.standard_skills = standard_skills
            q.embedding = EMBEDDING_MODEL.encode(", ".join(standard_skills))
            q.save()
            i += 1
