from django.db.models import Count, Q, IntegerField, Value
from django.db.models.functions import Coalesce

from recommender_core.matcher.base import BaseMatcherModel
from recommender_profile.models import UserProfile


class ListMatcher(BaseMatcherModel):
    pass