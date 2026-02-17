import asyncio
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .base import AIProvider


class HuggingFaceProvider(AIProvider):

    def __init__(self):

        # Select GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on: {self.device}")

        # T5 model for summarization (good balance of quality & speed)
        self.model_name = "t5-large"

        # Load tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

        # CPU stability
        if self.device.type == "cpu":
            torch.set_num_threads(4)

        # Warm up model
        self._run_summary(
            "T5 is a transformer model used for text summarization tasks.",
            None
        )

    def _run_summary(self, text: str, _):

        # T5 requires task prefix
        prefixed_text = "summarize: " + text

        # Tokenize input
        inputs = self.tokenizer(
            prefixed_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        # Dynamic length (safe + clean summaries)
        max_len = min(180, max(30, int(input_len * 0.4)))
        min_len = min(max_len - 1, max(20, int(max_len * 0.5)))

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_len,
                min_length=min_len,
                num_beams=4,                 # beam search = better summaries
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        # Decode to readable text
        summary = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return summary.strip()

    async def summarize(self, text: str, max_sentences: int = None):

        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(
            None,
            lambda: self._run_summary(text, max_sentences)
        )
