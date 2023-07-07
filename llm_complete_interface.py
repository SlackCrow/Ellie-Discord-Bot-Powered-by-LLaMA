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
    def __init__(self, model_name:str, model_path:str, n_ctx: int, n_gpu_layers:int=64):
        if not model_path.endswith('/'):
            model_path = model_path + '/'
        self.llm = Llama(model_path=model_path+model_name, n_ctx=n_ctx, seed=calculate_seed(), n_gpu_layers= n_gpu_layers)
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
            Only complete sentences for Assistant.
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
        self.llm = Llama(model_path=self.setup['model_path']+self.setup['model_name'], n_ctx=self.setup['n_ctx'], seed=calculate_seed(), n_batch=1024)

    def chat_completion(self, prompt_history: list, repeat_penalty:float, temp:float):
        self.count = self.count + 1
        output = self.llm(self.build_prompt(prompt_history), max_tokens=128, repeat_penalty=repeat_penalty, temperature=temp, stop=["###"], echo=False)
        if self.count == 10:
            self.reset_seed()
        return {'model': output['model'], 'usage': output['usage'], 'choices': [{'message':{'role': 'assistant', 'content': output['choices'][0]['text']}}]}
