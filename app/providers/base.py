from abc import ABC, abstractmethod

class AIProvider(ABC):

    @abstractmethod
    async def summarize(self, text: str, max_sentences: int) -> str:
        pass
