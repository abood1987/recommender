from rest_framework.generics import RetrieveUpdateDestroyAPIView
from rest_framework.mixins import CreateModelMixin

from recommender_profile.models import TaskProfile
from recommender_rest.serializer import TaskProfileSerializer


class TaskProfileCreateView(CreateModelMixin, RetrieveUpdateDestroyAPIView):
    queryset = TaskProfile.objects.all()
    serializer_class = TaskProfileSerializer