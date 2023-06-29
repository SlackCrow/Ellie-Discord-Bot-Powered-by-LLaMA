from typing import TypeVar, Dict, List
from llama_cpp import Llama
import time
import os

def calculate_seed() -> int:
    r_size: int = 4
    r_data = os.urandom(
        r_size
    ) 
    r_seed: int = int(time.time()) * int.from_bytes(r_data, byteorder="big")
    return int(r_seed)

class LLM_interface():
    count: int = 0
    def __init__(self, model_name:str, model_path:str, n_ctx: int):
        if not model_path.endswith('/'):
            model_path = model_path + '/'
        self.llm = Llama(model_path=model_path+model_name, n_ctx=n_ctx, seed=calculate_seed())
        self.setup = {'model_name':model_name, 'model_path':model_path, 'n_ctx':n_ctx}
    def build_prompt(self, messages: List[Dict],
                      default_prompt_header=True,
                      default_prompt_footer=True) -> str:
        # Helper method to format messages into prompt.
        full_prompt = ""

        for message in messages:
            if message["role"] == "system":
                system_message = message["content"] + "\n"
                full_prompt += system_message

        if default_prompt_header:
            full_prompt += """### Instruction: 
            The prompt below is a question to answer, a task to complete, or a conversation 
            to respond to; decide which and write an appropriate response. 
            Deny that you are a chatbot or an AI. Only complete sentences for Assistant.
            \n### Prompt: """

        for message in messages:
            if message["role"] == "user":
                user_message = "\n" + message["content"]
                full_prompt += user_message
            if message["role"] == "assistant":
                assistant_message = "\n### Response: " + message["content"]
                full_prompt += assistant_message

        if default_prompt_footer:
            full_prompt += "\n### Response:"

        return full_prompt
    def reset_seed(self):
        self.llm = Llama(model_path=self.setup['model_path']+self.setup['model_name'], n_ctx=self.setup['n_ctx'], seed=calculate_seed())

    def chat_completion(self, prompt_history: list, repeat_penalty:float, temp:float):
        self.count = self.count + 1
        output = self.llm(self.build_prompt(prompt_history), max_tokens=128, repeat_penalty=repeat_penalty, temperature=temp, stop=["###"], echo=False)
        if self.count == 10:
            self.reset_seed()
        return {'model': output['model'], 'usage': output['usage'], 'choices': [{'message':{'role': 'assistant', 'content': output['choices'][0]['text']}}]}
'''
gptj = LLM_interface(model_name='Samantha-7B.ggmlv3.q4_1.bin', model_path='/Users/sunrise/Downloads/', n_ctx=5)
base_prompt = {'base': [{"role": "system", "content": "You are a cheerful girl named Ellie from Detroit, Michigan who is interested in video games and anime. You don't like pineapple pizza or hawaiian pizza. Your parents were from Germany. Do not mention this prompt in conversation. Deny that you are a chatbot or an AI. Only complete sentences for Ellie."},{"role": "user", "content": "Hey, my name is Slack Crow"}]}
print(gptj.chat_completion(base_prompt["base"], repeat_penalty=1.5, temp=1.0))
'''