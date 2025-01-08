from django.urls import include, path, re_path
from recommender_rest import views


urlpatterns_v1 = [
    # path("token-auth/", views.ObtainAuthToken.as_view()),
    path("task/profile/", views.TaskProfileCreateView.as_view(), name="create_task_profile"),
    path("user/profile/", views.UserProfileCreateView.as_view(), name="create_user_profile"),
]

urlpatterns = [
    path("auth/", include("rest_framework.urls", namespace="rest_framework")),
    re_path(r"^(?P<version>(v1))/", include(urlpatterns_v1)),
]