import asyncio                     # For async execution
import torch                       # PyTorch for deep learning
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # HuggingFace tools
from .base import AIProvider       # Base class for AI providers


class HuggingFaceProvider(AIProvider):

    def __init__(self):

        # Select GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # Pretrained abstractive summarization model
        self.model_name = "facebook/bart-large-cnn"

        # Converts text into model tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Loads sequence-to-sequence generator model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        # Move model to CPU/GPU
        self.model.to(self.device)

        # Set model to inference mode
        self.model.eval()

        # Limit CPU threads for stability
        if self.device.type == "cpu":
            torch.set_num_threads(4)

        # Run once to preload model into memory
        self._run_summary("Warm up the model for initialization.", 3)

    def _run_summary(self, text: str, max_sentences: int):

        # Convert text into token tensors
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        # Disable gradient calculation (faster inference)
        with torch.no_grad():

            output_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_sentences * 50,     # controls size roughly by sentence count
                min_length=max_sentences * 20,

                num_beams=1,          # turn off beam copying
                do_sample=True,      # enable creativity
                temperature=1.15,     # higher = more new words
                top_p=0.9,           # nucleus sampling
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                early_stopping=True
            )


        # Convert tokens back to readable text
        return self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

    async def summarize(self, text: str, max_sentences: int):

        # Get async event loop
        loop = asyncio.get_running_loop()

        # Run heavy ML task in background thread
        return await loop.run_in_executor(
            None,
            lambda: self._run_summary(text, max_sentences)
        )
