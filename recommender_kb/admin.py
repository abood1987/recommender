from django.contrib import admin
from recommender_kb.models import SkillGroup, Occupation, Skill


class SemanticSearchAdmin(admin.ModelAdmin):
    def get_search_results(self, request, queryset, search_term):
        queryset, _ = super().get_search_results(request, queryset, search_term)
        if search_term:
            queryset = self.model.search(search_term)
        return queryset, _


@admin.register(SkillGroup)
class SkillGroupAdmin(SemanticSearchAdmin):
    list_display = ["label", "description"]
    list_filter = ["label", "description"]
    search_fields = ["label", "description"]


@admin.register(Occupation)
class OccupationAdmin(SemanticSearchAdmin):
    list_display = ["label", "description"]
    list_filter = ["label", "description"]
    search_fields = ["label", "description"]


@admin.register(Skill)
class SkillAdmin(SemanticSearchAdmin):
    list_display = ["label", "type", "description"]
    list_filter = ["label", "type", "description"]
    search_fields = ["label", "type", "description"]
