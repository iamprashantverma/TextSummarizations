import asyncio                     # Allows non-blocking async execution
import torch                       # PyTorch for tensor operations and GPU acceleration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # Hugging Face tools for NLP models
from .base import AIProvider       # Your base interface for AI providers

class HuggingFaceProvider(AIProvider):

    # Constructor: loads model, prepares device, and warms it up for faster first response
    def __init__(self):

        # Automatically choose GPU if available, otherwise fall back to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Print device so you know where computation is happening
        print(f"Running on: {self.device}")

        # Name of the pretrained T5 model (strong for summarization + fast enough)
        self.model_name = "t5-large"

        # Load tokenizer that converts text into model-readable tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load the actual neural network weights for summarization
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        # Move model to GPU or CPU memory
        self.model.to(self.device)

        # Put model in inference mode (disables training behavior like dropout)
        self.model.eval()

        # Limit CPU thread usage to avoid overloading system on inference
        if self.device.type == "cpu":
            torch.set_num_threads(4)

        # Run a dummy summary once so first real request isn't slow
        self._run_summary(
            "T5 is a transformer model used for text summarization tasks.",
            None
        )

    # Internal synchronous function that performs the actual summarization
    def _run_summary(self, text: str, _):

        # T5 requires a task prefix so it knows what job to perform
        prefixed_text = "summarize: " + text

        # Convert input text into token tensors for the model
        inputs = self.tokenizer(
            prefixed_text,              # Text with task instruction
            return_tensors="pt",        # Return PyTorch tensors
            truncation=True,            # Cut text if too long
            max_length=1024             # T5’s safe maximum input size
        ).to(self.device)               # Move tokens to GPU/CPU

        # Get number of tokens in input (used for dynamic summary sizing)
        input_len = inputs["input_ids"].shape[1]

        # Dynamically set summary length based on input size (clean & safe)
        max_len = min(180, max(30, int(input_len * 0.4)))   # Upper bound
        min_len = min(max_len - 1, max(20, int(max_len * 0.5)))  # Lower bound

        # Disable gradient calculations to save memory and speed up inference
        with torch.no_grad():

            # Generate summary tokens using beam search for quality
            output_ids = self.model.generate(
                inputs["input_ids"],

                max_length=max_len,
                min_length=min_len,

                do_sample=True,          # enables creativity
                temperature=1.1,        # higher = more new words
                top_p=0.9,              # nucleus sampling (smart randomness)

                num_beams=1,            # turn off strict beam search
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                early_stopping=True
            )


        # Convert output tokens back into readable text
        summary = self.tokenizer.decode(
            output_ids[0],                       # First (best) generated sequence
            skip_special_tokens=True,            # Remove model control tokens
            clean_up_tokenization_spaces=True    # Fix spacing artifacts
        )

        # Trim whitespace and return clean summary
        return summary.strip()

    # Public async method so summarization doesn't block the app
    async def summarize(self, text: str, max_sentences: int = None):

        # Get current async event loop
        loop = asyncio.get_running_loop()

        # Run heavy ML inference in a background thread
        return await loop.run_in_executor(
            None,                                # Default thread pool
            lambda: self._run_summary(text, max_sentences)  # Call sync method safely
        )
