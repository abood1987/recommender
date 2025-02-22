from django.urls import path, include
import recommender_test.views as views

urlpatterns_user_profile = [
    path("add/", views.AddUserProfileView.as_view(), name="add_user_profile"),
    path("<int:user_id>/update/", views.UpdateUserProfileView.as_view(), name="update_user_profile"),
    path("<int:user_id>/delete/", views.DeleteUserProfileView.as_view(), name="delete_user_profile"),
]

urlpatterns_task_profile = [
    path("add/", views.AddTaskProfileView.as_view(), name="add_task_profile"),
    path("<int:task_id>/update/", views.UpdateTaskProfileView.as_view(), name="update_task_profile"),
    path("<int:task_id>/delete/", views.DeleteTaskProfileView.as_view(), name="delete_task_profile"),
]

urlpatterns = [
    path("", views.FilteredTestCaseListView.as_view(), name="test_case_table"),
    path("add/", views.AddTestCaseView.as_view(), name="add_test_case"),
    path("<pk>/details/", views.TestCaseDetailsView.as_view(), name="test_case_details"),
    path("<pk>/update/", views.UpdateTestCaseView.as_view(), name="edit_test_case"),
    path("<pk>/delete/", views.DeleteTestCaseView.as_view(), name="delete_test_case"),
    path("<pk>/start/", views.StartTestCaseView.as_view(), name="start_test_case"),

    path("<int:pk>/user/", include(urlpatterns_user_profile)),
    path("<int:pk>/task/", include(urlpatterns_task_profile)),
]