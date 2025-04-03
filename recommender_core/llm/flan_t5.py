from recommender_core.llm.base import LLMModelBase
from pathlib import Path
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from recommender_core.utils.collector import ClassTracer


class FlanT5Model(LLMModelBase):
    @ClassTracer.exclude
    def _get_model(self):
        if self.model_path:
            path = Path(self.model_path)
            if path.exists() and any(path.iterdir()):
                model_name = self.model_path
            else:
                model_name = self.model_name
        else:
            model_name = self.model_name
        return (
            T5Tokenizer.from_pretrained(model_name),
            T5ForConditionalGeneration.from_pretrained(model_name)
        )

    def prompt(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():  # No training, just predictions => Disables gradient tracking (Faster & Memory Efficient)
            outputs = self.model.generate(inputs.input_ids, max_length=512, num_beams=5, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


