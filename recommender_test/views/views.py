from bootstrap_modal_forms.generic import (
    BSModalCreateView,
    BSModalUpdateView,
    BSModalDeleteView
)
from django.urls import reverse_lazy
from recommender_profile.models import UserProfile, TaskProfile
from recommender_test.forms import UserProfileForm, TaskProfileForm


class TestBaseMixin:
    def get_success_url(self):
        return reverse_lazy("test_case_details", kwargs={"pk": self.kwargs["pk"]})

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs["test_case_id"] = self.kwargs["pk"]
        return kwargs


class AddUserProfileView(TestBaseMixin, BSModalCreateView):
    model = UserProfile
    form_class = UserProfileForm
    success_message = "Success: user was created."
    template_name = "recommender_test/modal_form_template.html"
    extra_context = {
        "form_title": "Create User",
    }


class UpdateUserProfileView(TestBaseMixin, BSModalUpdateView):
    model = UserProfile
    pk_url_kwarg = "user_id"
    form_class = UserProfileForm
    success_message = "Success: user was updated."
    template_name = "recommender_test/modal_form_template.html"
    extra_context = {
        "form_title": "Update User",
    }


class DeleteUserProfileView(BSModalDeleteView):
    model = UserProfile
    pk_url_kwarg = "user_id"
    template_name = "recommender_test/delete_template.html"
    success_message = "Success: user was deleted."
    extra_context = {
        "form_title": "Delete User",
    }

    def get_success_url(self):
        return reverse_lazy("test_case_details", kwargs={"pk": self.kwargs["pk"]})


class AddTaskProfileView(TestBaseMixin, BSModalCreateView):
    model = TaskProfile
    form_class = TaskProfileForm
    success_message = "Success: task was created."
    template_name = "recommender_test/modal_form_template.html"
    extra_context = {
        "form_title": "Create Task",
    }


class UpdateTaskProfileView(TestBaseMixin, BSModalUpdateView):
    model = TaskProfile
    pk_url_kwarg = "task_id"
    form_class = TaskProfileForm
    success_message = "Success: task was updated."
    template_name = "recommender_test/modal_form_template.html"
    extra_context = {
        "form_title": "Update Task",
    }


class DeleteTaskProfileView(BSModalDeleteView):
    model = TaskProfile
    pk_url_kwarg = "task_id"
    template_name = "recommender_test/delete_template.html"
    success_message = "Success: task was deleted."
    extra_context = {
        "form_title": "Delete Task",
    }

    def get_success_url(self):
        return reverse_lazy("test_case_details", kwargs={"pk": self.kwargs["pk"]})
