from django.db import models
from django_jsonform.models.fields import ArrayField

from recommender_profile.models import UserProfile, TaskProfile


# class TestAddress(models.Model):
#     country = models.CharField(max_length=100, default="Austria")
#     state = models.CharField(max_length=100)
#     city = models.CharField(max_length=100)
#     zip = models.IntegerField()
#
#     def __str__(self):
#         return f"{self.city}, {self.state}, {self.country} - {self.zip}"

class TestCase(models.Model):
    name = models.CharField(max_length=100)
    users = models.ManyToManyField(UserProfile)
    tasks = models.ManyToManyField(TaskProfile)
#
# class TestUser(models.Model):
#     address = models.ForeignKey(TestAddress, on_delete=models.PROTECT)
#     test_case = models.ForeignKey(TestCase, on_delete=models.CASCADE, related_name='test_users')
#
#
# class TestTask(models.Model):
#     title = models.CharField(max_length=255)
#     description = models.TextField()
#     address = models.ForeignKey(TestAddress, on_delete=models.PROTECT)
#     test_case = models.ForeignKey(TestCase, on_delete=models.CASCADE, related_name='test_tasks')
#     skills = ArrayField(models.CharField(max_length=50), size=10)
#
#
# class TestSkill(models.Model):
#     test_user = models.ForeignKey(TestUser, on_delete=models.CASCADE, related_name="user_skills", null=True)
#     test_task = models.ForeignKey(TestTask, on_delete=models.CASCADE, related_name="task_skills", null=True)
#     skill = models.CharField(max_length=512)