from abc import ABC, abstractmethod
from llms.base import LLM

QUERY_REFLECTION_PROMPT = """
You are helpful assistant of hospital.
Below is the conversation history and the current message from the user.

### Conversation History:
{history}  
### Current Message:
{query}

## Requirements:
* Reformat the user's current message using information from the conversation history if necessary, to ensure that submitting it to the information retrieval system yields the best result.
* The response must consist only of the reformatted current message, with no additional explanations.
* If the current message is already suitable for direct submission to the information retrieval system, return it unchanged.
"""
    
class QueryReflection:
    def __init__(self, llm: LLM):
        self.llm = llm
        
    def _concat_and_format_texts(self, data):
        concatenatedTexts = []
        for entry in data:
            role = entry.get('role', '')
            content = entry.get("content", "")
            concatenatedTexts.append(f"{role}: {content} \n")
        return ''.join(concatenatedTexts)
        
    def __call__(self, chatHistory, lastItemsConsidereds=5):
        print("RUNNING QUERY REFLECTION . . .")
        
        if len(chatHistory) >= lastItemsConsidereds:
            chatHistory = chatHistory[len(chatHistory) - lastItemsConsidereds:]

        historyString = self._concat_and_format_texts(chatHistory[:-1])
        query = chatHistory[-1]["content"]

        higherLevelSummariesPrompt = QUERY_REFLECTION_PROMPT.replace("{history}", historyString).replace("{query}", query)

        completion = self.llm.generate_content(higherLevelSummariesPrompt)
        print("RESULT: ", completion)

        return completion