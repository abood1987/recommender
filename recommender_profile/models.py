from django.contrib.postgres.fields import ArrayField
from django.db import models
from pgvector.django import HnswIndex

from recommender_core.models import BaseVectorModel


class Address(models.Model):
    country = models.CharField(max_length=100, default="Austria")
    City = models.CharField(max_length=100)
    zip = models.IntegerField()


class UserProfileManager(models.Manager):
    def active(self):
        return self.get_queryset().filter(is_active=True)


class UserProfile(BaseVectorModel):
    # date & time & experiences & skills level & old

    objects = UserProfileManager()
    external_id = models.PositiveBigIntegerField(unique=True)
    skills = ArrayField(models.TextField())
    standard_skills = ArrayField(models.CharField(max_length=255), null=True)

    is_available = models.BooleanField(default=True)
    address = models.ForeignKey(Address, on_delete=models.PROTECT)

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
    # date & time & requirements & old

    objects = TaskProfileManager()

    external_id = models.PositiveBigIntegerField()
    deception = models.TextField()

    is_available = models.BooleanField(default=True)
    title = models.CharField(max_length=255)
    standard_title = models.CharField(max_length=255, null=True)
    # responsibilities = ArrayField(models.TextField())
    # benefits = ArrayField(models.TextField())
    skills = ArrayField(models.TextField())
    standard_skills = ArrayField(models.CharField(max_length=255), null=True)
    address = models.ForeignKey(Address, on_delete=models.PROTECT)

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