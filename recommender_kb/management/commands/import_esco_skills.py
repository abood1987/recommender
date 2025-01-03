import os
import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from recommender_kb.models import (
    Occupation,
    Skill,
    ISCOGroup,
)


class Command(BaseCommand):
    help = "Import ESCO skills & occupations from csv"

    def add_arguments(self, parser):
        parser.add_argument("--path", type=str, required=True, help="Container path where ISCO CSV files are located.")
        parser.add_argument("--clear", action="store_true", help="Delete old data before importing")
        parser.add_argument("--ignore-occ", action="store_true", dest="ignore_occupations", help="ignore the import of the occupations.")
        parser.add_argument("--ignore-skills", action="store_true", dest="ignore_skills", help="ignore the import of the skills.")
        parser.add_argument("--ignore-isco-groups", action="store_true", dest="ignore_isco_groups", help="ignore the import of the isco groups.")
        parser.add_argument("--ignore-rel-skills", action="store_true", dest="ignore_rel_skills", help="ignore the import of the skills relations.")
        parser.add_argument("--ignore-rel-occ-skills", action="store_true", dest="ignore_rel_occ_skills", help="ignore the import of the occupations-skills relations.")

    def handle(self, *args, **options):
        path = options['path']
        # Check if the path exists
        if not os.path.exists(path):
            raise CommandError(f"The specified path does not exist: {path}")

        self.stdout.write("---START---")
        if options.get("clear", False):
            self.clear_data()
            self.stdout.write("---Old data cleared----")

        if not options.get("ignore_isco_groups", False):
            self.import_isco_groups(path)
            self.stdout.write("---ISCO groups imported----")

        if not options.get("ignore_occupations", False):
            self.import_occupations(path)
            self.stdout.write("---Occupations imported----")

        if not options.get("ignore_skills", False):
            self.import_skills(path)
            self.stdout.write("---Skills imported----")

        if not options.get("ignore_rel_skills", False):
            self.import_skills_relations(path)
            self.stdout.write("---Skills-Skills relations imported----")

        if not options.get("ignore_rel_occ_skills", False):
            self.import_occupations_skills_relations(path)
            self.stdout.write("---occupations-Skills relations imported----")
        self.stdout.write("---END---")

    def get_file_path(self, path: str, file_name: str) -> str:
        file_path = os.path.join(path, file_name)
        if not os.path.isfile(file_path):
            raise CommandError(f"The specified file does not exist: {file_path}")
        return file_path

    def get_dataframe(self, path: str, file_name: str, dtype=None) -> pd.DataFrame:
        df = pd.read_csv(self.get_file_path(path, file_name), dtype=dtype)
        df = df.astype(str)
        # Check if the DataFrame is empty
        if df.empty:
            raise CommandError(f"The file is empty: {file_name}")
        return df

    def get_instance_dict(self, row, csv_map: dict) -> dict:
        instance_dict = {}
        for key, value in csv_map.items():
            if isinstance(value, str):
                instance_dict[value] = row[key]
            elif isinstance(value, dict):
                instance_dict[value["field"]] = value["function"](row[key])
        return instance_dict


    def clear_data(self):
        Occupation.objects.all().delete()
        Skill.objects.all().delete()
        ISCOGroup.objects.all().delete()

    def import_isco_groups(self, path: str):
        csv_map = {
            "conceptUri": "uri",
            "code": "code",
            "preferredLabel": "label",
            "description": "description",
        }
        embedding_fields = ["preferredLabel", "description"]
        df = self.get_dataframe(path, "ISCOGroups_en.csv", dtype={"code": str})
        for index, row in df.iterrows():
            ISCOGroup.objects.create(**self.get_instance_dict(row, csv_map))

    def import_occupations(self, path: str):
        csv_map = {
            "conceptUri": "uri",
            "preferredLabel": "label",
            "description": "description",
            "iscoGroup": {
                "field": "isco_group",
                "function": lambda code: ISCOGroup.objects.get(code=code) if code else None
            },
            "altLabels": {
                "field": "alt_labels",
                "function": lambda val: val.splitlines()
            },
            "hiddenLabels": {
                "field": "hidden_labels",
                "function": lambda val: val.splitlines()
            }
        }
        embedding_fields = ["preferredLabel", "altLabels", "hiddenLabels", "description"]
        df = self.get_dataframe(path, "occupations_en.csv", dtype={"iscoGroup": str})
        for index, row in df.iterrows():
            Occupation.objects.create(**self.get_instance_dict(row, csv_map))

    def import_skills(self, path: str):
        csv_map = {
            "conceptUri": "uri",
            "skillType": {
                "field": "type",
                "function": lambda val: next((key for key, value in Skill.SKILL_TYPE_CHOICES if value == val), None)
            },
            "preferredLabel": "label",
            "description": "description",
            "altLabels": {
                "field": "alt_labels",
                "function": lambda val: val.splitlines()
            },
            "hiddenLabels": {
                "field": "hidden_labels",
                "function": lambda val: val.splitlines()
            }
        }
        embedding_fields = ["preferredLabel", "altLabels", "hiddenLabels", "description"]
        df = self.get_dataframe(path, "skills_en.csv")
        for index, row in df.iterrows():
            Skill.objects.create(**self.get_instance_dict(row, csv_map))

    def import_skills_relations(self, path: str):
        df = self.get_dataframe(path, "skillSkillRelations_en.csv")
        for index, row in df.iterrows():
            s = Skill.objects.get(uri=row["originalSkillUri"])
            s.related_skills.add(Skill.objects.get(uri=row["relatedSkillUri"]), through_defaults={"type": row["relationType"]})

    def import_occupations_skills_relations(self, path: str):
        df = self.get_dataframe(path, "occupationSkillRelations_en.csv")
        for index, row in df.iterrows():
            o = Occupation.objects.get(uri=row["occupationUri"])
            o.skills.add(Skill.objects.get(uri=row["skillUri"]), through_defaults={"type": row["relationType"]})

"""
# The best hybrid database for your use case is Weaviate.
Why Weaviate?

    Built-in Vector and Graph Capabilities:
        Weaviate natively combines vector search with a semantic graph, making it ideal for storing relationships between entities (skills, occupations, knowledge) and querying them efficiently.

    Ease of Use:
        It has a simple API and supports querying both relationships and vectors seamlessly using GraphQL and REST.

    Scalability:
        Designed to handle large datasets with both relational and vector-based queries.

    Customizable Embeddings:
        Supports integration with various embedding models (e.g., OpenAI, HuggingFace) to vectorize your data directly.

    Active Development and Community:
        Regular updates, strong documentation, and an active user community make it a future-proof choice.



# https://ec.europa.eu/esco/api/search ? language=en&type=occupation&text=manager (offset=0 & limit = 100)

# https://ec.europa.eu/esco/api/resource/occupation?uri=some occupation url & language=en

# https://ec.europa.eu/esco/api/resource/related
# https://data.europa.eu/esco/concept-scheme/occupations&relation=hasTopConcept&language=en&full=true
"""
