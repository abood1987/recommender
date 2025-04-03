import ast
import os
import random
from email.policy import default

import pandas as pd
from django.core import checks
from django.core.management.base import BaseCommand, CommandError
from recommender_profile.models import UserProfile, TaskProfile, Address


class Command(BaseCommand):
    help = "Import external data"
    requires_system_checks = []

    def check(self, app_configs=None, tags=None, display_num_errors=False, include_deployment_checks=False,
              fail_level=checks.ERROR, databases=None):
        self.stdout.write(self.style.WARNING("'Import external data': SKIPPING SYSTEM CHECKS!\n"))

    def add_arguments(self, parser):
        parser.add_argument("--path", type=str, required=True, help="Container path where external CSV files are located.")
        parser.add_argument("--clear", dest="clear", action="store_true", help="Delete old data before importing")

    def handle(self, *args, **options):
        self.path = options["path"]
        if not os.path.exists(self.path):
            raise CommandError(f"The specified path does not exist: {self.path}")

        self.stdout.write("---START---")
        if options.get("clear", False):
            UserProfile.objects.all().delete()
            TaskProfile.objects.all().delete()
            self.stdout.write("---Old data cleared----")

        self.import_occupations()
        self.stdout.write("---Occupations imported----")

        self.import_skills()
        self.stdout.write("---Skills imported----")
        self.stdout.write("---END---")

    def get_file_path(self, file_name: str) -> str:
        file_path = os.path.join(self.path, file_name)
        if not os.path.isfile(file_path):
            raise CommandError(f"The specified file does not exist: {file_path}")
        return file_path

    def get_dataframe(self, file_name: str, dtype=None) -> pd.DataFrame:
        df = pd.read_csv(self.get_file_path(file_name))
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
                instance_dict[value["field"]] = value["function"](row)
        return instance_dict

    def _get_address(self, row):
        if "city" in row and "state" in row and "country" in row and "zip" in row:
            address, _ = Address.objects.get_or_create(
                city=row["city"],
                country=row["country"],
                zip=row["zip"],
                state=row["state"]
            )
            return address
        else:
            return Address.objects.get(id=random.randint(1, Address.objects.count()))

    def import_skills(self):
        csv_map = {
            # "id": "external_id",
            "skills": "skills",
            "address": {
                "field": "address",
                "function": self._get_address,
            }
        }
        df = self.get_dataframe("user_profiles.csv")
        df["skills"] = df["skills"].apply(ast.literal_eval)
        for index, row in df.iterrows():
            UserProfile.objects.create(**{**self.get_instance_dict(row, csv_map), **{"external_id": index}})

    def import_occupations(self):
        csv_map = {
            "title": "title",
            "description": {
                "field": "description",
                "function": lambda csv_row: csv_row.get("description", "default description"),
            },
            "requirements": "skills",
            "address": {
                "field": "address",
                "function": self._get_address,
            }
        }
        df = self.get_dataframe("job_profiles.csv")
        df["requirements"] = df["requirements"].apply(ast.literal_eval)
        for index, row in df.iterrows():
            TaskProfile.objects.create(**{**self.get_instance_dict(row, csv_map), **{"external_id": index}})
