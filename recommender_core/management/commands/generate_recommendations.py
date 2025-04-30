from django.core.management.base import BaseCommand
from django.db.models import QuerySet

from recommender_core.utils.helper import get_matcher
from recommender_profile.models import TaskProfile, UserProfile, Recommendation


class Command(BaseCommand):
    help = "Generate Recommendations"

    def handle(self, *args, **options):
        self.stdout.write("---START---")
        tasks = TaskProfile.objects.all()
        users = UserProfile.objects.all()

        filtered_users, filtered_tasks = self.get_filtered_objects(users, tasks)
        matcher = get_matcher()
        recommendations = matcher.get_recommendations(filtered_users, filtered_tasks)
        for user_id, task_ids in recommendations.items():
            users.get(id=user_id).recommendations.set(tasks.filter(id__in=task_ids))

        self.stdout.write("---END---")

    def get_filtered_objects(self, users: QuerySet[UserProfile], tasks: QuerySet[TaskProfile]) -> tuple[QuerySet[UserProfile], QuerySet[TaskProfile]]:
        last_update = None
        last_recommendation = Recommendation.objects.order_by("modified_at").last()
        if last_recommendation:
            last_update = last_recommendation.modified_at

        if last_update:
            filtered_tasks = tasks.filter(modified_at__gte=last_update)
            filtered_users = users.filter(modified_at__gte=last_update)

            if filtered_tasks.exists() and not filtered_users.exists():
                # rematch filtered tasks with all users
                filtered_users = users
            elif not filtered_tasks.exists() and filtered_users.exists():
                # rematch filtered users with all tasks
                filtered_tasks = tasks
            elif filtered_tasks.exists() and filtered_users.exists():
                # start the match for all objects
                filtered_tasks = tasks
                filtered_users = users
            else:
                # no change since last update
                pass
        else:
            # start the match for all objects
            filtered_tasks = tasks
            filtered_users = users
        return filtered_users, filtered_tasks