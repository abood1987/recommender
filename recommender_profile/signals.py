from django.db.models.signals import post_save
from django.dispatch import receiver

from recommender_profile.models import TaskProfile, UserProfile


@receiver(post_save, sender=TaskProfile)
def task_profile_post_save(sender, instance, created, **kwargs):
    if created:
        instance.kb_matching_and_generate_embedding()
    else:
        update_fields = kwargs.get("update_fields")
        if update_fields and ("title" in update_fields or "skills" in update_fields):
            instance.kb_matching_and_generate_embedding()


@receiver(post_save, sender=UserProfile)
def task_profile_post_save(sender, instance, created, **kwargs):
    if created:
        instance.kb_matching_and_generate_embedding()
    else:
        update_fields = kwargs.get("update_fields")
        if update_fields and "skills" in update_fields:
            instance.kb_matching_and_generate_embedding()