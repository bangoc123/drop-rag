from abc import ABC, abstractmethod



class LLM(ABC):
    @abstractmethod
    def create_agentic_chunker_message(self, system_prompt, messages, max_tokens=1000, temperature=1.0):
        """
        Create message method for using agentic chunker
        """
        
    @abstractmethod
    def generate_content(self, prompt: str):
        """Generate content with given prompt"""