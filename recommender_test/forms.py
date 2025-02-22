from bootstrap_modal_forms.forms import BSModalModelForm
from django import forms
from django_jsonform.forms.fields import ArrayFormField

from recommender_profile.models import UserProfile, TaskProfile
from .models import TestCase


class TestCaseForm(BSModalModelForm):
    class Meta:
        model = TestCase
        fields = ["name"]


class ProfileFormBase(BSModalModelForm):
    skills = ArrayFormField(forms.CharField(max_length=512))
    external_id = forms.IntegerField(widget=forms.HiddenInput(), required=False)

    def __init__(self, *args, **kwargs) -> None:
        self.test_case_id = kwargs.pop("test_case_id")
        super().__init__(*args, **kwargs)
        self.is_adding = self.instance._state.adding

        for field_name, field in self.fields.items():
            if isinstance(field.widget, forms.Textarea):
                field.widget.attrs.update({'rows': 2})

    def clean(self):
        cleaned_data = super().clean()
        print(cleaned_data)
        if self.is_adding:
            last_profile = self.Meta.model.objects.all().order_by("external_id").last()
            cleaned_data["external_id"] = last_profile.external_id + 1 if last_profile else 0
        return cleaned_data

    def save(self, commit=True):
        instance = super().save(False)
        instance.skills = self.cleaned_data["skills"]
        instance.save()
        return instance


class UserProfileForm(ProfileFormBase):
    class Meta:
        model = UserProfile
        fields = ["address", "skills", "external_id"]

    def save(self, commit=True):
        instance = super().save(commit)
        if self.is_adding:
            TestCase.objects.get(id=self.test_case_id).users.add(instance)
        return instance


class TaskProfileForm(ProfileFormBase):
    class Meta:
        model = TaskProfile
        fields = ["title", "description", "address", "skills", "external_id"]

    def save(self, commit=True):
        instance = super().save(commit)
        if self.is_adding:
            TestCase.objects.get(id=self.test_case_id).tasks.add(instance)
        return instance
