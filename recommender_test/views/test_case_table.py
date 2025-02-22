from bootstrap_modal_forms.generic import BSModalCreateView, BSModalUpdateView, BSModalDeleteView
from django.urls import reverse_lazy
from django.views.generic import DetailView
from django_filters import FilterSet
from django_filters.views import FilterView
from django_tables2 import tables, A, LinkColumn, TemplateColumn
from django_tables2.views import SingleTableMixin

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

    def post(self, request, *args, **kwargs):
        return super().post(request, *args, **kwargs)
