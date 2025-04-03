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

from recommender.settings import VECTOR_SETTINGS, EXTRACTOR_SIMPLE, EXTRACTOR_NER_JOBBERT, EXTRACTOR_NER_ESCOXLMR, \
    EXTRACTOR_LLM, LLM_FLAN_T5, LLM_FLAN_T5_FT, EXTRACTOR_SPLIT
from recommender_core.utils.collector import DataCollector, ClassTracer
from recommender_core.utils.helper import get_extractor
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
        "methods": {
            "simple": "Simple",
            "split": "Split",
            "jobbert": "NER",
            "escoxlmr": "Escoxlmr",
            "flat_t5": "Flat T5",
            "flat_t5_ft": "Flat T5 (fine tuning)"
        }
    }
    METHODS_MAP = {
        "simple": {**VECTOR_SETTINGS, "extractor": EXTRACTOR_SIMPLE},
        "split": {**VECTOR_SETTINGS, "extractor": EXTRACTOR_SPLIT},
        "jobbert": {**VECTOR_SETTINGS, "extractor": EXTRACTOR_NER_JOBBERT},
        "escoxlmr": {**VECTOR_SETTINGS, "extractor": EXTRACTOR_NER_ESCOXLMR},
        "flat_t5": {**VECTOR_SETTINGS, "extractor": EXTRACTOR_LLM, "llm": LLM_FLAN_T5},
        "flat_t5_ft": {**VECTOR_SETTINGS, "extractor": EXTRACTOR_LLM, "llm": LLM_FLAN_T5_FT},
    }

    def dispatch(self, request, *args, **kwargs):
        ClassTracer.active = True
        DataCollector().data.clear()
        return super().dispatch(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        instance = self.get_object()
        method = request.POST.get("select_method")
        extractor = get_extractor(json.dumps(self.METHODS_MAP[method]))

        with ThreadPoolExecutor() as executor:
            # Run user and task processing in parallel
            futures = []
            for u in instance.users.all():
                futures.append(executor.submit(u.generate_standard_skills_and_embedding, extractor))

            for s in instance.tasks.all():
                futures.append(executor.submit(s.generate_standard_skills_and_embedding, extractor))

            for future in futures: # Wait for all tasks to finish
                future.result()

        self.template_name = "recommender_test/test_case_results.html"
        return self.render_to_response({
            "object": instance,
            "traces": DataCollector().data
        })
