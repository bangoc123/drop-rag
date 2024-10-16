from constant import GEMINI
import google.generativeai as genai

class OnlineLLMs:
    def __init__(self, name, api_key=None):
        self.name = name
        self.model = None

        # Configure and initialize the Gemini model if the name is "gemini" and an API key is provided
        if self.name.lower() == "gemini" and api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-pro')

    def set_model(self, model):
        """Set the LLM model for this instance."""
        self.model = model

    def generate_content(self, prompt):
        """Generate content using the specified LLM model."""
        if not self.model:
            raise ValueError("Model is not set. Please set a model using set_model().")

        # Generate content based on the model type
        if self.name.lower() == "gemini":
            # Handle Gemini response structure
            response = self.model.generate_content(prompt)
            try:
                # Extract the text from the nested response structure
                content = response.candidates[0].content.parts[0].text
            except (IndexError, AttributeError):
                raise ValueError("Failed to parse the Gemini response structure.")
        else:
            raise ValueError(f"Unknown model name: {self.name}")

        # Ensure the content is a string
        if not isinstance(content, str):
            content = str(content)

        return content


