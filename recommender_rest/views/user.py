from rest_framework.generics import RetrieveUpdateDestroyAPIView
from rest_framework.mixins import CreateModelMixin

from recommender_profile.models import UserProfile
from recommender_rest.serializer import UserProfileSerializer


class UserProfileCreateView(CreateModelMixin, RetrieveUpdateDestroyAPIView):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer