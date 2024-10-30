from constant import GEMINI, OPENAI
import google.generativeai as genai
from openai import OpenAI
from .base import LLM
import backoff

class OnlineLLMs(LLM):
    def __init__(self, name, api_key=None, model_version=None):
        self.name = name
        self.model = None
        self.model_version = model_version

        # Configure and initialize the Gemini model if the name is "gemini" and an API key is provided
        if self.name.lower() == GEMINI and api_key:
            genai.configure(
                api_key=api_key
            )
            self.model = genai.GenerativeModel(
                model_name=model_version
            )
        elif self.name.lower() == OPENAI and api_key:
            self.model = OpenAI(
                api_key=api_key,
            )

    def set_model(self, model):
        """Set the LLM model for this instance."""
        self.model = model
    
    def parse_message(self, messages):
        mapping = {
            "user": "user",
            "assistant": "model"
        }
        return [
            {"role": mapping[mess["role"]], "parts": mess["content"]}
            for mess in messages
        ]
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def create_agentic_chunker_message(self, system_prompt, messages, max_tokens=1000, temperature=1):
        if self.name.lower() == "gemini":
            try:
                messages = self.parse_message(messages)
                response = self.model.generate_content(
                    [
                        {"role": "user", "parts": system_prompt},
                        {"role": "model", "parts": "I understand. I will strictly follow your instruction!"},
                        *messages
                    ],
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature
                    )
                )
                return response.text
            except Exception as e:
                print(f"Error occurred: {e}, retrying...")
                raise e   
        elif self.name.lower() == "openai":
            try:
                gpt_messages = [
                    {"role": "system", "content": system_prompt}
                ] + messages

                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    messages=gpt_messages,
                    temperature=temperature
                )

                return completion.choices[0].message.content
            except Exception as e:
                print(f"Error occurred: {e}, retrying...")
                raise e
        else:
            raise ValueError(f"Unknown model name: {self.name}")

    def generate_content(self, prompt):
        """Generate content using the specified LLM model."""
        if not self.model:
            raise ValueError("Model is not set. Please set a model using set_model().")

        # Generate content based on the model type
        if self.name.lower() == GEMINI:
            # Handle Gemini response structure
            response = self.model.generate_content(prompt)
            try:
                # Extract the text from the nested response structure
                content = response.candidates[0].content.parts[0].text
            except (IndexError, AttributeError):
                raise ValueError("Failed to parse the Gemini response structure.")
        elif self.name.lower() == OPENAI:
            # Handle OpenAI response structure
            response = self.model.chat.completions.create(
                model=self.model_version,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            try:
                # Extract the text from the nested response structure
                content = response.choices[0].message.content
            except (IndexError, AttributeError):
                raise ValueError("Failed to parse the OpenAI response structure.")    
        else:
            raise ValueError(f"Unknown model name: {self.name}")

        # Ensure the content is a string
        if not isinstance(content, str):
            content = str(content)

        return content


