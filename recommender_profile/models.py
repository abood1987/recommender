from django.contrib.postgres.fields import ArrayField
from django.db import models
from django.db.models import QuerySet
from django.db.models.functions import Coalesce
from pgvector.django import HnswIndex

from recommender_core.extractor.base import ExtractorBase
from recommender_core.models import BaseVectorModel, BaseModel
from recommender_core.utils.helper import get_extractor
from recommender_kb.models import Skill, Occupation


class Address(BaseModel):
    country = models.CharField(max_length=100, default="Austria")
    state = models.CharField(max_length=100)
    city = models.CharField(max_length=100)
    zip = models.IntegerField()

    def __str__(self):
        return f"{self.country}, {self.state}, {self.city}, {self.zip}"


class UserProfileManager(models.Manager):
    def active(self):
        return self.get_queryset().filter(is_available=True)


class UserProfile(BaseVectorModel):
    # date & time & experiences & skills level & old

    objects = UserProfileManager()
    external_id = models.PositiveBigIntegerField(unique=True)
    skills = ArrayField(models.TextField())
    standard_skills = models.ManyToManyField(Skill, related_name="user_profiles")

    is_available = models.BooleanField(default=True)
    address = models.ForeignKey(Address, on_delete=models.PROTECT)
    recommendations = models.ManyToManyField("TaskProfile", through="Recommendation",  related_name="recommendations")

    def generate_standard_skills_and_embedding(self, extractor: ExtractorBase|None = None):
        extractor = extractor or get_extractor()
        self.standard_skills.set(extractor.extract_skills(self.skills))
        self.embedding = extractor.embedding_model.encode(
            ", ".join(list(self.standard_skills.values_list("label", flat=True)) or [])
        )
        self.save()

    class Meta:
        indexes = [
            HnswIndex(
                name=f"hnsw_user_embedding_index",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            )
        ]

    def __str__(self):
        return f"User {self.id}"


class TaskProfileManager(models.Manager):
    def active(self):
        return self.get_queryset().filter(is_available=True)


class TaskProfile(BaseVectorModel):
    # date & time & requirements & responsibilities &  benefits & old

    objects = TaskProfileManager()

    external_id = models.PositiveBigIntegerField()
    description = models.TextField()

    is_available = models.BooleanField(default=True)
    title = models.CharField(max_length=255)
    skills = ArrayField(models.TextField(), null=True)
    standard_title = models.ForeignKey(Occupation, null=True, related_name="task_profiles", on_delete=models.PROTECT)
    standard_skills = models.ManyToManyField(Skill, related_name="task_profiles")
    address = models.ForeignKey(Address, on_delete=models.PROTECT)

    def generate_standard_skills_and_embedding(self, extractor: ExtractorBase|None = None):
        extractor = extractor or get_extractor()
        self.standard_title = extractor.extract_occupation(self.title).first()
        self.standard_skills.set(extractor.extract_skills(self.skills))
        self.embedding = extractor.embedding_model.encode(
            ", ".join(list(self.standard_skills.values_list("label", flat=True)) or []))
        self.save()

    class Meta:
        indexes = [
            HnswIndex(
                name=f"hnsw_company_embedding_index",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            )
        ]

    def __str__(self):
        return f"{self.title}"


class Recommendation(BaseModel):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    task_profile = models.ForeignKey(TaskProfile, on_delete=models.CASCADE)