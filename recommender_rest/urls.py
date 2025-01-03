from django.urls import include, path, re_path
from recommender_rest import views


urlpatterns_v1 = [
    # path("token-auth/", views.ObtainAuthToken.as_view()),
]

urlpatterns = [
    path("auth/", include("rest_framework.urls", namespace="rest_framework")),
    re_path(r"^(?P<version>(v1))/", include(urlpatterns_v1)),
]