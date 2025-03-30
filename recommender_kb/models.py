from django.contrib.postgres.fields import ArrayField
from django.db import models
from pgvector.django import HnswIndex

from recommender_core.models import BaseVectorModel


class SkillGroup(BaseVectorModel):
    uri = models.URLField(unique=True)
    label = models.CharField(max_length=255)
    description = models.TextField()

    class Meta:
        # https://medium.com/@simeon.emanuilov/integrating-a-vector-database-into-django-using-pgvector-72322b9debbe
        indexes = [
            HnswIndex(
                name=f"hnsw_isco_group_embedding_index",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            )
        ]

    def __str__(self):
        return self.label


class Occupation(BaseVectorModel):
    # preferredLabel + altLabels + hiddenLabels
    uri = models.URLField(unique=True)
    label = models.CharField(max_length=255)
    description = models.TextField()
    skills = models.ManyToManyField("Skill", through="Occupation2Skill", related_name='occupations')
    alt_labels = ArrayField(models.CharField(max_length=255), blank=True, null=True)
    hidden_labels = ArrayField(models.CharField(max_length=200), blank=True, null=True)

    def get_related_skills(self):
        return list(self.skills.values_list("label", flat=True))

    class Meta:
        indexes = [
            HnswIndex(
                name=f"hnsw_occupation_embedding_index",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            )
        ]

    def __str__(self):
        return self.label


class Skill(BaseVectorModel):
    SKILL_TYPE_SKILL = "skill/competence"
    SKILL_TYPE_KNOWLEDGE = "knowledge"
    SKILL_TYPE_CHOICES = (
        (SKILL_TYPE_SKILL, "Skill/Competence"),
        (SKILL_TYPE_KNOWLEDGE, "Knowledge"),
    )

    # preferredLabel + altLabels + hiddenLabels
    uri = models.URLField(unique=True)
    label = models.CharField(max_length=255)
    type = models.CharField(max_length=20, choices=SKILL_TYPE_CHOICES, null=True, blank=True)
    description = models.TextField()
    related_skills = models.ManyToManyField("Skill", through="Skill2Skill", related_name='original_skills')
    alt_labels = ArrayField(models.CharField(max_length=255), blank=True, null=True)
    hidden_labels = ArrayField(models.CharField(max_length=200), blank=True, null=True)
    broader = models.ForeignKey(SkillGroup, on_delete=models.PROTECT, null=True, blank=True, related_name='skills')

    class Meta:
        indexes = [
            HnswIndex(
                name=f"hnsw_skill_embedding_index",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            )
        ]

    def __str__(self):
        return self.label


class Occupation2Skill(models.Model):
    ESSENTIAL = "essential"
    OPTIONAL = "optional"

    RELATION_TYPE_CHOICES = [
        (ESSENTIAL, "Essential"),
        (OPTIONAL, "Optional"),
    ]

    occupation = models.ForeignKey(Occupation, on_delete=models.CASCADE, related_name='occupation')
    skill = models.ForeignKey(Skill, on_delete=models.CASCADE, related_name='skill')
    type = models.CharField(max_length=20, choices=RELATION_TYPE_CHOICES)

    def __str__(self):
        return f"{self.occupation} -> {self.skill} ({self.type})"


class Skill2Skill(models.Model):
    ESSENTIAL = "essential"
    OPTIONAL = "optional"

    RELATION_TYPE_CHOICES = [
        (ESSENTIAL, "Essential"),
        (OPTIONAL, "Optional"),
    ]

    original_skill = models.ForeignKey(Skill, on_delete=models.CASCADE, related_name="original_skill")
    related_skill = models.ForeignKey(Skill, on_delete=models.CASCADE, related_name="related_skill")
    type = models.CharField(max_length=20, choices=RELATION_TYPE_CHOICES)

    def __str__(self):
        return f"{self.original_skill} -> {self.related_skill} ({self.type})"
