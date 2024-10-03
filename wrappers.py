import os
import litellm
from dotenv import load_dotenv
from openai import OpenAI


class LiteLLMWrapper:
    '''
    Wrapper around liteLLM, gives you access to multiple LLMs (100+ LLMs) 
    '''
    def __init__(self,):
        load_dotenv()
    def chat_completion(self,**kwargs):
        response = litellm.completion(**kwargs)
        if kwargs["stream"]:
            return response
        return response.choices[0].message.content
    
class OpenAIWrapper:
    '''
    model:int
    temperature:int, 
    prompt:str, 
    max_tokens:int,
    stream:bool,
    top_p:float 
    ''' 
    def __init__(self):
        load_dotenv()
        self.__api_key = os.getenv("OPENAI_API_KEY")
        self.__api_base = "https://api.openai.com/v1/"
        self.client = OpenAI(api_key=self.__api_key,base_url=self.__api_base)

    def chat_completion(self,**kwargs):
        response = self.client.chat.completions.create(**kwargs)
        if kwargs.get("stream", False):
            return response#generator object 
        return response.choices[0].message.content

    def completion(self,**kwargs):
        response = self.client.completions.create(**kwargs)
        if kwargs.get("stream", False):
            return response# generator object 
        return response.choices[0].text

class TogetherAPIWrapper(OpenAIWrapper):
    '''
    model:int
    temperature:int, 
    prompt:str, 
    max_tokens:int,
    stream:bool,
    top_p:float 
    ''' 
    load_dotenv()
    def __init__(self):
        super(TogetherAPIWrapper).__init__()
        self.__api_key = os.getenv("TOGETHER_API_KEY")
        self.__api_base = "https://api.together.xyz/v1"
        self.client = OpenAI(api_key=self.__api_key,base_url=self.__api_base)
