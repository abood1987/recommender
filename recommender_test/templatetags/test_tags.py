from django import template
register = template.Library()


@register.simple_tag
def get_recommendation(task, users):
    return task.get_recommendations(users)