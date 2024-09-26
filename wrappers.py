import os
import openai 
from dotenv import load_dotenv

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
        #initialize key and base 
        openai.api_key = self.__api_key
        openai.api_base = self.__api_base

    def chat_completion(self,**kwargs):
        response = openai.ChatCompletion.create(**kwargs)
        if kwargs.get("stream", False):
            return response#generator object 
        return response["choices"][0]["message"]["content"]

    def completion(self,**kwargs):
        response = openai.Completion.create(**kwargs)
        if kwargs.get("stream", False):
            return response# generator object 
        return response["choices"][0]["text"]
    
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
        openai.api_key = self.__api_key
        openai.api_base = self.__api_base


