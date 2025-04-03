import os
import pandas as pd
from django.core import checks
from django.core.management.base import BaseCommand, CommandError
from nltk.sem.chat80 import borders

from recommender_kb.models import (
    Occupation,
    Skill,
    SkillGroup,
)


class Command(BaseCommand):
    help = "Import ESCO skills & occupations from csv"
    requires_system_checks = []

    def check(self, app_configs=None, tags=None, display_num_errors=False, include_deployment_checks=False,
              fail_level=checks.ERROR, databases=None):
        self.stdout.write(self.style.WARNING("'Import skills & occupations from csv': SKIPPING SYSTEM CHECKS!\n"))

    def add_arguments(self, parser):
        parser.add_argument("--path", type=str, required=True, help="Container path where ISCO CSV files are located.")
        parser.add_argument("--clear", action="store_true", help="Delete old data before importing")
        parser.add_argument("--broader", action="store_true", help="Import broader data")

    def handle(self, *args, **options):
        path = options['path']
        # Check if the path exists
        if not os.path.exists(path):
            raise CommandError(f"The specified path does not exist: {path}")

        self.stdout.write("---START---")
        if options.get("broader", False):
            self.import_broader_rel(path)
        else:
            if options.get("clear", False):
                self.clear_data()
                self.stdout.write("---Old data cleared----")

            self.import_groups(path)
            self.stdout.write("---ISCO groups imported----")

            self.import_occupations(path)
            self.stdout.write("---Occupations imported----")

            self.import_skills(path)
            self.stdout.write("---Skills imported----")

            self.import_skills_relations(path)
            self.stdout.write("---Skills-Skills relations imported----")

            self.import_occupations_skills_relations(path)
            self.stdout.write("---occupations-Skills relations imported----")

            self.import_broader_rel(path)
            self.stdout.write("---broader relations imported----")
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
        SkillGroup.objects.all().delete()

    def import_groups(self, path: str):
        csv_map = {
            "conceptUri": "uri",
            "preferredLabel": "label",
            "description": "description",
        }
        df = self.get_dataframe(path, "skillGroups_en.csv", dtype={"code": str})
        for index, row in df.iterrows():
            SkillGroup.objects.create(**self.get_instance_dict(row, csv_map))

    def import_occupations(self, path: str):
        csv_map = {
            "conceptUri": "uri",
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

    def import_broader_rel(self, path: str):
        df = self.get_dataframe(path, "broaderRelationsSkillPillar_en.csv", dtype={"code": str})
        for index, row in df.iterrows():
            skills = Skill.objects.filter(uri=row["conceptUri"])
            if not skills.exists():
                print("skills", row["conceptUri"])
                continue
            skill = skills.get()
            broader = SkillGroup.objects.filter(uri=row["broaderUri"])
            if broader.exists():
                broader = broader.get()
            else:
                s_broader = Skill.objects.filter(uri=row["broaderUri"])
                if s_broader.exists():
                    s_broader = s_broader.get()
                    broader, _ = SkillGroup.objects.get_or_create(label=s_broader.label, uri=s_broader.uri, description=s_broader.description)
                else:
                    print("skills", skill.label, "broaderUri", row["broaderUri"])
                    continue
            skill.broader = broader
            skill.save()