from django.contrib import admin

from recommender_core.utils.helper import model_semantic_search
from recommender_kb.models import ISCOGroup, Occupation, Skill


class SemanticSearchAdmin(admin.ModelAdmin):
    def get_search_results(self, request, queryset, search_term):
        queryset, _ = super().get_search_results(request, queryset, search_term)
        if search_term:
            queryset = model_semantic_search(self.model, search_term)
        return queryset, _


@admin.register(ISCOGroup)
class ISCOGroupAdmin(SemanticSearchAdmin):
    list_display = ["label", "description"]
    list_filter = ["label", "description"]
    search_fields = ["label", "description"]


@admin.register(Occupation)
class ISCOGroupAdmin(SemanticSearchAdmin):
    list_display = ["label", "description"]
    list_filter = ["label", "description"]
    search_fields = ["label", "description"]


@admin.register(Skill)
class ISCOGroupAdmin(SemanticSearchAdmin):
    list_display = ["label", "type", "description"]
    list_filter = ["label", "type", "description"]
    search_fields = ["label", "type", "description"]