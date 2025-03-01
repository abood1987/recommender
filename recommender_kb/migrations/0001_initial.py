# Generated by Django 4.2.17 on 2024-12-27 19:21

import django.contrib.postgres.fields
from django.db import migrations, models
import django.db.models.deletion
import pgvector.django.indexes
import pgvector.django.vector


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ISCOGroup',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('embedding', pgvector.django.vector.VectorField(dimensions=768, null=True)),
                ('uri', models.URLField(unique=True)),
                ('code', models.CharField(max_length=20, unique=True)),
                ('label', models.CharField(max_length=255)),
                ('description', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='Occupation',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('embedding', pgvector.django.vector.VectorField(dimensions=768, null=True)),
                ('uri', models.URLField(unique=True)),
                ('label', models.CharField(max_length=255)),
                ('description', models.TextField()),
                ('alt_labels', django.contrib.postgres.fields.ArrayField(base_field=models.CharField(max_length=255), blank=True, null=True, size=None)),
                ('hidden_labels', django.contrib.postgres.fields.ArrayField(base_field=models.CharField(max_length=200), blank=True, null=True, size=None)),
                ('isco_group', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.PROTECT, related_name='occupations', to='recommender_kb.iscogroup')),
            ],
        ),
        migrations.CreateModel(
            name='Skill',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('embedding', pgvector.django.vector.VectorField(dimensions=768, null=True)),
                ('uri', models.URLField(unique=True)),
                ('label', models.CharField(max_length=255)),
                ('type', models.CharField(blank=True, choices=[('skill/competence', 'Skill/Competence'), ('knowledge', 'Knowledge')], max_length=20, null=True)),
                ('description', models.TextField()),
                ('alt_labels', django.contrib.postgres.fields.ArrayField(base_field=models.CharField(max_length=255), blank=True, null=True, size=None)),
                ('hidden_labels', django.contrib.postgres.fields.ArrayField(base_field=models.CharField(max_length=200), blank=True, null=True, size=None)),
            ],
        ),
        migrations.CreateModel(
            name='Skill2Skill',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('type', models.CharField(choices=[('essential', 'Essential'), ('optional', 'Optional')], max_length=20)),
                ('original_skill', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='original_skill', to='recommender_kb.skill')),
                ('related_skill', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='related_skill', to='recommender_kb.skill')),
            ],
        ),
        migrations.AddField(
            model_name='skill',
            name='related_skills',
            field=models.ManyToManyField(related_name='original_skills', through='recommender_kb.Skill2Skill', to='recommender_kb.skill'),
        ),
        migrations.CreateModel(
            name='Occupation2Skill',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('type', models.CharField(choices=[('essential', 'Essential'), ('optional', 'Optional')], max_length=20)),
                ('occupation', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='occupation', to='recommender_kb.occupation')),
                ('skill', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='skill', to='recommender_kb.skill')),
            ],
        ),
        migrations.AddField(
            model_name='occupation',
            name='skills',
            field=models.ManyToManyField(related_name='occupations', through='recommender_kb.Occupation2Skill', to='recommender_kb.skill'),
        ),
        migrations.AddIndex(
            model_name='iscogroup',
            index=pgvector.django.indexes.HnswIndex(ef_construction=64, fields=['embedding'], m=16, name='hnsw_isco_group_embedding_index', opclasses=['vector_cosine_ops']),
        ),
        migrations.AddIndex(
            model_name='skill',
            index=pgvector.django.indexes.HnswIndex(ef_construction=64, fields=['embedding'], m=16, name='hnsw_skill_embedding_index', opclasses=['vector_cosine_ops']),
        ),
        migrations.AddIndex(
            model_name='occupation',
            index=pgvector.django.indexes.HnswIndex(ef_construction=64, fields=['embedding'], m=16, name='hnsw_occupation_embedding_index', opclasses=['vector_cosine_ops']),
        ),
    ]
