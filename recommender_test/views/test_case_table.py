from concurrent.futures import ThreadPoolExecutor
import json
from bootstrap_modal_forms.generic import (
    BSModalCreateView,
    BSModalUpdateView,
    BSModalDeleteView,
    BSModalReadView
)
from django.urls import reverse_lazy
from django.views.generic import DetailView
from django_filters import FilterSet
from django_filters.views import FilterView
from django_tables2 import tables, A, LinkColumn, TemplateColumn
from django_tables2.views import SingleTableMixin

from recommender.settings import VECTOR_SETTINGS, EXTRACTOR_EMBEDDING_SIMILARITY, EXTRACTOR_NER_JOBBERT, \
    EXTRACTOR_NER_ESCOXLMR, \
    EXTRACTOR_LLM, LLM_FLAN_T5, LLM_FLAN_T5_FT, EXTRACTOR_SPLIT, MATCHER_BINARY_VECTOR, MATCHER_EMBEDDINGS, \
    MATCHER_FUZZY, MATCHER_OVERLAP, MATCHER_TF_IDF, EXTRACTOR_HYBRID
from recommender_core.utils.collector import DataCollector, ClassTracer
from recommender_core.utils.helper import get_extractor, get_matcher
from recommender_profile.models import TaskProfile, UserProfile
from recommender_test.forms import TestCaseForm
from recommender_test.models import TestCase


class TestCaseTable(tables.Table):
    name = LinkColumn("test_case_details", args=[A("pk")])
    actions = TemplateColumn(verbose_name="", template_name='recommender_test/test_case_table_actions.html',)
    class Meta:
        model = TestCase
        fields = ("name",)


class TestCaseFilter(FilterSet):
    class Meta:
        model = TestCase
        fields = {"name": ["contains"]}


class FilteredTestCaseListView(SingleTableMixin, FilterView):
    table_class = TestCaseTable
    model = TestCase
    template_name = "recommender_test/test_case_table_template.html"
    filterset_class = TestCaseFilter


class TestCaseDetailsView(DetailView):
    model = TestCase
    template_name = "recommender_test/test_case_details.html"

class AddTestCaseView(BSModalCreateView):
    model = TestCase
    form_class = TestCaseForm
    template_name = "recommender_test/modal_form_template.html"
    success_url = reverse_lazy("test_case_table")
    extra_context = {
        "form_title": "Create Test Case",
    }

class UpdateTestCaseView(BSModalUpdateView):
    model = TestCase
    form_class = TestCaseForm
    template_name = "recommender_test/modal_form_template.html"
    success_url = reverse_lazy("test_case_table")
    extra_context = {
        "form_title": "Update Test Case",
    }


class DeleteTestCaseView(BSModalDeleteView):
    model = TestCase
    template_name = "recommender_test/delete_template.html"
    success_url = reverse_lazy("test_case_table")
    success_message = "Success"
    extra_context = {
        "form_title": "Delete Test Case",
    }


class StartTestCaseView(BSModalReadView):
    model = TestCase
    template_name = "recommender_test/test_case_start.html"
    extra_context = {
        "btn_text": "Start",
        "form_title": "Start Test Case",
        "extractor_methods": {
            "simple": "Simple",
            "split": "Split",
            "jobbert": "NER",
            "escoxlmr": "Escoxlmr",
            "flat_t5": "Flat T5",
            "flat_t5_ft": "Flat T5 (fine tuning)",
            "hybrid": "Hybrid (Simple + LLM)"
        },
        "matcher_methods": {
            "binary_vector": "Binary Vector",
            "embeddings": "Embeddings",
            "fuzzy": "Fuzzy",
            "overlap": "Overlap",
            "tfidf": "TF-IDF"
        }
    }
    EXTRACTOR_METHODS_MAP = {
        "simple": EXTRACTOR_EMBEDDING_SIMILARITY,
        "split": EXTRACTOR_SPLIT,
        "jobbert": EXTRACTOR_NER_JOBBERT,
        "escoxlmr": EXTRACTOR_NER_ESCOXLMR,
        "flat_t5": {"extractor": EXTRACTOR_LLM, "llm": LLM_FLAN_T5},
        "flat_t5_ft": {"extractor": EXTRACTOR_LLM, "llm": LLM_FLAN_T5_FT},
        "hybrid": EXTRACTOR_HYBRID
    }
    MATCHER_METHODS_MAP = {
        "binary_vector": MATCHER_BINARY_VECTOR,
        "embeddings": MATCHER_EMBEDDINGS,
        "fuzzy": MATCHER_FUZZY,
        "overlap": MATCHER_OVERLAP,
        "tfidf": MATCHER_TF_IDF
    }

    def dispatch(self, request, *args, **kwargs):
        ClassTracer.active = True
        DataCollector().data.clear()
        return super().dispatch(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        instance = self.get_object()
        extractor_configs = self.EXTRACTOR_METHODS_MAP[request.POST.get("extractor_method")]
        matcher_configs = self.MATCHER_METHODS_MAP[request.POST.get("matcher_method")]
        threshold_value = request.POST.get("threshold_value")
        fuzzy_threshold_value = request.POST.get("fuzzy_threshold_value")
        include_broader = bool(request.POST.get("include_broader", False))
        top_k_matching = int(request.POST.get("top_k_matching"))

        matcher_configs["configuration"].update({
            "threshold": float(threshold_value),
            "fuzzy_threshold": float(fuzzy_threshold_value),
            "include_broader": include_broader,
            "top_k": top_k_matching
        })

        configs = {
            **VECTOR_SETTINGS,
            **(extractor_configs if isinstance(extractor_configs, dict) else {"extractor": extractor_configs}),
            "matcher": matcher_configs
        }
        extractor = get_extractor(json.dumps(configs))
        matcher = get_matcher(json.dumps(configs))

        tasks = instance.tasks.all()
        users = instance.users.all()
        # tasks = TaskProfile.objects.filter(testcase__in=list(TestCase.objects.values_list("id", flat=True)))
        # users = UserProfile.objects.filter(testcase__in=list(TestCase.objects.values_list("id", flat=True)))
        with ThreadPoolExecutor() as executor:
            # Run user and task processing in parallel
            futures = []
            for u in users:
                futures.append(executor.submit(u.generate_standard_skills_and_embedding, extractor))

            for s in tasks:
                futures.append(executor.submit(s.generate_standard_skills_and_embedding, extractor))

            for future in futures: # Wait for all tasks to finish
                future.result()

        recommendations = matcher.get_recommendations(users, tasks)
        self.template_name = "recommender_test/test_case_results.html"
        recommendations_dict = {
            users.get(id=u_id): tasks.filter(id__in=t_ids)
            for u_id, t_ids in recommendations.items()
        }
        return self.render_to_response({
            "object": instance,
            "traces": DataCollector().data,
            "recommendations": recommendations_dict,
            "recommendations_map": {
                "threshold": float(threshold_value),
                "fuzzy_threshold": float(fuzzy_threshold_value),
                "include_broader": include_broader,
                "top_k": top_k_matching,
                "Users count": len(recommendations_dict.keys()),
                "Tasks count": sum(qs.count() for qs in recommendations_dict.values()),
                "Matched users count": len([u for u, t in recommendations_dict.items() if t.count() > 0]),
                "Matched tasks count": len({task.id for qs in recommendations_dict.values() for task in qs}),
                "Recommendations": {
                    f"User: {user.id}": {
                        "ids": list(tasks.values_list("id", flat=True)),
                        "titles": list(tasks.values_list("title", flat=True))
                    } for user, tasks in recommendations_dict.items() if tasks.exists()
                }
            },
        })
