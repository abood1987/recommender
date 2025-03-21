from recommender_core.llm.base import LLMModelBase
from pathlib import Path
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from recommender_core.utils.collector import ClassTracer


class FlanT5Model(LLMModelBase):
    @ClassTracer.exclude
    def _get_model(self):
        if not self.model_path:
            return (
                T5Tokenizer.from_pretrained(self.model_name),
                T5ForConditionalGeneration.from_pretrained(self.model_name)
            )

        path = Path(self.model_path)
        if path.exists() and any(path.iterdir()):
            return (
                T5Tokenizer.from_pretrained(self.model_path),
                T5ForConditionalGeneration.from_pretrained(self.model_path)
            )

    def prompt(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():  # No training, just predictions => Disables gradient tracking (Faster & Memory Efficient)
            outputs = self.model.generate(inputs.input_ids, max_length=512, num_beams=5, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


