from typing import Any

from recommender_core.utils.singleton import Singleton


class DataCollector(Singleton):
    def __init__(self):
        self.data: dict[str, Any] = {}