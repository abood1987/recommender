import os
import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from recommender_profile.models import UserProfile, CompanyProfile


class Command(BaseCommand):
    help = "Import skills & occupations from csv"

    def add_arguments(self, parser):
        parser.add_argument("--path", type=str, required=True, help="Container path where external CSV files are located.")
        parser.add_argument("--clear-skills", dest="clear_skills", action="store_true", help="Delete old data before importing")
        parser.add_argument("--clear-occ", dest="clear_occ", action="store_true", help="Delete old data before importing")
        parser.add_argument("--ignore-occ", action="store_true", dest="ignore_occupations", help="ignore the import of the occupations.")
        parser.add_argument("--ignore-skills", action="store_true", dest="ignore_skills", help="ignore the import of the skills.")
        parser.add_argument("--sep", default=";", type=str)
        parser.add_argument("--quote", default='"', type=str)

    def handle(self, *args, **options):
        self.sep = options["sep"]
        self.quote = options["quote"]
        self.path = options["path"]
        if not os.path.exists(self.path):
            raise CommandError(f"The specified path does not exist: {self.path}")

        self.stdout.write("---START---")
        if options.get("clear_skills", False):
            UserProfile.objects.all().delete()
            self.stdout.write("---Skills data cleared----")

        if options.get("clear_occ", False):
            CompanyProfile.objects.all().delete()
            self.stdout.write("---Occupations data cleared----")

        if not options.get("ignore_occupations", False):
            self.import_occupations()
            self.stdout.write("---Occupations imported----")

        if not options.get("ignore_skills", False):
            self.import_skills()
            self.stdout.write("---Skills imported----")
        self.stdout.write("---END---")

    def get_file_path(self, file_name: str) -> str:
        file_path = os.path.join(self.path, file_name)
        if not os.path.isfile(file_path):
            raise CommandError(f"The specified file does not exist: {file_path}")
        return file_path

    def get_dataframe(self, file_name: str, dtype=None) -> pd.DataFrame:
        df = pd.read_csv(self.get_file_path(file_name), sep=self.sep, dtype=dtype, quotechar=self.quote)
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

    def import_skills(self):
        csv_map = {
            # "id": "external_id",
            "Skills": {
                "field": "skills",
                "function": lambda vals: vals.split(","),
            },
        }
        df = self.get_dataframe("external_skills.csv")
        for index, row in df.iterrows():
            UserProfile.objects.create(**{**self.get_instance_dict(row, csv_map), **{"external_id": index}})

    def import_occupations(self):
        csv_map = {
            "id": "external_id",
            "content": "occupation"
        }
        df = self.get_dataframe("external_occupations.csv")
        for index, row in df.iterrows():
            CompanyProfile.objects.create(**self.get_instance_dict(row, csv_map))
