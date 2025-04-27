import unicodedata
import re
from django.db.models.signals import post_save
from django.dispatch import receiver

from recommender_profile.models import TaskProfile, UserProfile


def normalize_text(text):
    # 1. Unicode normalization (NFKC is often preferred for general use)
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove extra spaces and normalize punctuation spacing
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces into one
    text = re.sub(r"\s([?.!,:;])", r"\1", text)  # remove space before punctuation
    text = text.strip()  # remove leading/trailing whitespace
    return text


def normalize_skills(skills):
    normalized_skills = []
    for skill in skills:
        normalized_skills.append(normalize_text(skill))
    return normalized_skills


@receiver(post_save, sender=TaskProfile)
def task_profile_post_save(sender, instance, created, **kwargs):
    if created:
        instance.skills = normalize_skills(instance.skills)
        instance.title = normalize_text(instance.title)
        instance.save()
        instance.generate_standard_skills_and_embedding()
    else:
        update_fields = kwargs.get("update_fields")
        if update_fields and ("title" in update_fields or "skills" in update_fields):
            if "title" in update_fields:
                instance.title = normalize_text(instance.title)
            if "skills" in update_fields:
                instance.skills = normalize_skills(instance.skills)
            instance.save()
            instance.generate_standard_skills_and_embedding()


@receiver(post_save, sender=UserProfile)
def user_profile_post_save(sender, instance, created, **kwargs):
    if created:
        instance.skills = normalize_skills(instance.skills)
        instance.save()
        instance.generate_standard_skills_and_embedding()
    else:
        update_fields = kwargs.get("update_fields")
        if update_fields and "skills" in update_fields:
            instance.skills = normalize_skills(instance.skills)
            instance.save()
            instance.generate_standard_skills_and_embedding()