from rest_framework.generics import RetrieveUpdateDestroyAPIView
from rest_framework.mixins import CreateModelMixin

from recommender_profile.models import UserProfile
from recommender_rest.serializer import UserProfileSerializer


class UserProfileView(CreateModelMixin, RetrieveUpdateDestroyAPIView):
    lookup_field = "external_id"
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer