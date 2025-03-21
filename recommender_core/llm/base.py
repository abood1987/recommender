from abc import ABC
from transformers import PreTrainedTokenizerBase
from transformers.modeling_utils import SpecificPreTrainedModelType
from recommender_core.utils.collector import ClassTracer, DataCollector
from recommender_core.utils.singleton import Singleton


class LLMModelBase(Singleton, ABC, metaclass=ClassTracer):

    @ClassTracer.exclude
    def __init__(self, model_name: str, model_path: str | None = None, **kwargs):
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer, self.model = self._get_model()
        self.collector = DataCollector()

    @ClassTracer.exclude
    def _get_model(self) -> tuple[bool | PreTrainedTokenizerBase, SpecificPreTrainedModelType]:
        raise NotImplementedError

    def prompt(self, prompt: str) -> str:
        """
        Start direct prompt using selected model.
        """
        raise NotImplementedError
