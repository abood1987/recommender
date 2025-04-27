from rest_framework.generics import RetrieveUpdateDestroyAPIView
from rest_framework.mixins import CreateModelMixin

from recommender_profile.models import TaskProfile
from recommender_rest.serializer import TaskProfileSerializer


class TaskProfileView(CreateModelMixin, RetrieveUpdateDestroyAPIView):
    lookup_field = "external_id"
    queryset = TaskProfile.objects.all()
    serializer_class = TaskProfileSerializer