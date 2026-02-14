import asyncio
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .base import AIProvider


class HuggingFaceProvider(AIProvider):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = "sshleifer/distilbart-cnn-12-6"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        self.model.to(self.device)
        self.model.eval()

        if self.device.type == "cpu":
            torch.set_num_threads(4)

    def _run_summary(self, text: str, max_sentences: int):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        max_len = max_sentences * 40
        min_len = max(60, max_len // 2)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_len,
                min_length=min_len,
                num_beams=2,           
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        return self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

    async def summarize(self, text: str, max_sentences: int):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._run_summary(text, max_sentences)
        )
