from django.contrib.postgres.fields import ArrayField
from django.db import models
from django.db.models.functions import Coalesce
from pgvector.django import HnswIndex

from recommender_core.extractor.base import BaseExtractorModel
from recommender_core.models import BaseVectorModel
from recommender_core.utils.helper import get_llm_model
from recommender_kb.models import Skill, Occupation


class Address(models.Model):
    country = models.CharField(max_length=100, default="Austria")
    state = models.CharField(max_length=100)
    city = models.CharField(max_length=100)
    zip = models.IntegerField()

    def __str__(self):
        return f"{self.country}, {self.state}, {self.city}, {self.zip}"


class UserProfileManager(models.Manager):
    def active(self):
        return self.get_queryset().filter(is_active=True)


class UserProfile(BaseVectorModel):
    # date & time & experiences & skills level & old

    objects = UserProfileManager()
    external_id = models.PositiveBigIntegerField(unique=True)
    skills = ArrayField(models.TextField())
    standard_skills = models.ManyToManyField(Skill, related_name="user_profiles")

    is_available = models.BooleanField(default=True)
    address = models.ForeignKey(Address, on_delete=models.PROTECT)

    def generate_standard_skills_and_embedding(self, llm_model: BaseExtractorModel = None):
        llm_model = llm_model or get_llm_model()
        self.standard_skills.set(*llm_model.get_standard_skills(self.skills))
        self.embedding = llm_model.encode(
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
        return f"User {self.id} --> {self.external_id}"


class TaskProfileManager(models.Manager):
    def active(self):
        return self.get_queryset().filter(is_active=True)


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

    def generate_standard_skills_and_embedding(self, llm_model: BaseExtractorModel = None):
        llm_model = llm_model or get_llm_model()
        self.standard_title = llm_model.get_standard_occupation(self.title)
        self.standard_skills.set(*llm_model.get_standard_skills(self.skills))
        self.embedding = llm_model.encode(
            ", ".join(list(self.standard_skills.values_list("label", flat=True)) or []))
        self.save()

    def get_recommendations(self):
        return UserProfile.objects.filter(
            standard_skills__overlap=self.standard_skills
        ).annotate(
            overlap_count=Coalesce(
                models.Count(
                    "standard_skills",
                    filter=models.Q(standard_skills__overlap=self.standard_skills)
                ), models.Value(0, output_field=models.IntegerField())
            )
        ).order_by("-overlap_count")

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