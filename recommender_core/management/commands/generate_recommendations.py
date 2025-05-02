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

    def get_filtered_objects(
            self,
            users: QuerySet[UserProfile],
            tasks: QuerySet[TaskProfile]
    ) -> tuple[QuerySet[UserProfile], QuerySet[TaskProfile]]:

        last_recommendation = Recommendation.objects.order_by("modified_at").last()
        if not last_recommendation:
            return users, tasks

        last_update = last_recommendation.modified_at
        updated_tasks = tasks.filter(modified_at__gte=last_update)
        updated_users = users.filter(modified_at__gte=last_update)

        if updated_tasks.exists() and not updated_users.exists():
            # rematch filtered tasks with all users
            return users, updated_tasks
        elif not updated_tasks.exists() and updated_users.exists():
            # rematch filtered users with all tasks
            return updated_users, tasks
        elif updated_tasks.exists() and updated_users.exists():
            # start the match for all objects
            return users, tasks
        else:
            # No updates → nothing to match
            return QuerySet.none(users), QuerySet.none(tasks)
