import openai
import os
import time 
import json
from openai import OpenAI

class ChatGPT:

    def __init__(self, model_name):
        
        self.model_name = model_name
        self.prompt_token=0
        self.gen_token=0
        self.cost=0
         
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.sleep_time = 0.5
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            )
        
        self.prompt_token_cost=0
        self.gen_token_cost=0
        
        if(self.model_name=="gpt-3.5-turbo-0613"):
            self.prompt_token_cost=1.5
            self.gen_token_cost=2

        elif(self.model_name=="gpt-3.5-turbo-1106"):
            self.prompt_token_cost=1
            self.gen_token_cost=2

        elif(self.model_name=="gpt-3.5-turbo-0125"):
            self.prompt_token_cost=0.5
            self.gen_token_cost=1.5

        elif(self.model_name=="gpt-4"):
            self.prompt_token_cost=30
            self.gen_token_cost=60
        
        elif(self.model_name=="gpt-4-0613"):
            self.prompt_token_cost=30
            self.gen_token_cost=60

        elif(self.model_name=="gpt-4-0125-preview"):
            self.prompt_token_cost=10
            self.gen_token_cost=30

        elif(self.model_name=="gpt-4-1106-preview"):
            self.prompt_token_cost=10
            self.gen_token_cost=30

        elif(self.model_name=="gpt-4-vision-preview"):
            self.prompt_token_cost=10
            self.gen_token_cost=30
            

    def generate(self,prompt,sys_prompt=None,temperature=1.0):

        message_list = []

        if(sys_prompt is not None):
            message_list.append({"role": "system", "content": sys_prompt})

        message_list.append({"role": "user", "content": prompt})

        try:
            response_object = self.client.chat.completions.create(
                model=self.model_name,
                messages=message_list,
                temperature = temperature
                )
        except:
            time.sleep(self.sleep_time)
            response_object = self.client.chat.completions.create(
                model=self.model_name,
                messages=message_list,
                temperature = self.temperature
                )

        response = response_object.choices[0].message.content
        self.prompt_token+=response_object.usage.prompt_tokens
        self.gen_token+=response_object.usage.completion_tokens
        
        return response
    
    def get_cost(self):
        
        cost = (self.prompt_token_cost*self.prompt_token+self.gen_token_cost*self.gen_token)/1000000

        print("Prompt Token Consumption",self.prompt_token)
        print("Gen Token Consumption",self.gen_token)
        
        return cost
