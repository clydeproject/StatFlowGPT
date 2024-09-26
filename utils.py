import re 
import io
import os 
import json 
import contextlib
import streamlit as st 
import matplotlib.pyplot as plt 
from prompt_templates import REPHRASE_PROMPT,GENERATE_CODE,ERROR_CORRECT,SUMMARIZE_RESULTS

class SimpleCodeExecutor:
    def __init__(self,environment,**safelocals):
        self.__safe_locals = safelocals  # External variables
        self.stdout_buffer = io.StringIO()  # Capture output
        self.stderr_buffer = io.StringIO()  # Capture errors
        self.environment = environment 

    def run(self,code):
        try:
            with contextlib.redirect_stdout(self.stdout_buffer), contextlib.redirect_stderr(self.stderr_buffer):
                exec(code, {}, self.__safe_locals)
                if self.environment == "st":
                    if plt.get_fignums():
                        st.pyplot(plt.gcf())
                        plt.close()  
        except Exception as e:
            return {"success": False, "output": self.stdout_buffer.getvalue(), "error": str(e)}
        return {"success": True, "output": self.stdout_buffer.getvalue(), "error": self.stderr_buffer.getvalue()}

class Analyst:
    def __init__(self,df,client,model=None,code_executor=SimpleCodeExecutor):
        self.df = df 
        self.client = client 
        self.model = model 
        self.code_executor = code_executor(environment="st",df=self.df)#run dynamically generated code locally

    def rephrase_query(self,query,model):
        prompt = REPHRASE_PROMPT.format(
            dfpreview=self.df.head(),
            colnames = self.df.columns,
            userquery = query,
            )
        
        rephrased_query = self.client.completion(
            model=model,temperature=0,max_tokens=256,
            messages=[{"role":"user","content":prompt},],
            )
        return rephrased_query
    
    def generate_code(self,query,model):
        prompt = GENERATE_CODE.format(
            nrows=self.df.shape[0],
            ncols=self.df.shape[1],
            colnames=self.df.columns,
            dfhead=self.df.head(),
            query=query,
        )

        code = self.client.chat_completion(
            model=model,temperature=0,max_tokens=1000,
            messages = [{"role":"user","content":prompt}]
        )
        return code 

    def error_correct_code(self,query,code,error_message,model):
        prompt = ERROR_CORRECT.format(
            query=query,
            error_code=code,
            error_message=error_message,
            nrows=self.df.shape[0],
            ncols=self.df.shape[1],
            colnames=self.df.columns.tolist(),
            dfhead=self.df.head()
        ) 

        error_correction = self.client.chat_completion(
            model=model,temperature=0,max_tokens=1000,
            messages= [{"role":"user","content":prompt}]
        )
        return error_correction

    def summarize_results(self,query,results,model,**kwargs):
        prompt = SUMMARIZE_RESULTS.format(
            query=query,
            results=results
        )
        summary = self.client.chat_completion(
            stream=kwargs["stream"],
            model=model,temperature=0,max_tokens=1000,
            messages = [{"role":"user","content":prompt}],
        )
        return summary
    
    def extract_code_from_generation(self,generation):
        code_blocks = re.findall(r'```(.*?)```', generation, re.DOTALL)
        python_code_blocks = [block.strip()[7:] for block in code_blocks if block.startswith('python')]
        return '\n'.join(python_code_blocks) if python_code_blocks else generation

    def run_chain(self,query):
        if self.model is None:
            raise ValueError(f"To use run_chain() an LLM model has to be chosen, please choose a model while instantiating Analyst()")
        
        #query answering chain 
        rephrased_query = self.rephrase_query(query,model=self.model)
        print(rephrased_query)#debug
        generated_code = self.generate_code(query=rephrased_query,model=self.model)
        print(generated_code)#debug
        #state of the execution of generated code is the following format for SimpleCode Executor
        # {'success': Bool, 'output': Str, 'error': Bool}
        for _ in range(3):#3 tries to fix code 
            state = self.code_executor.run(self.extract_code_from_generation(generated_code))
            print(state)#debug
            if state["success"]:
                return state["output"]
            
            corrected_code = self.error_correct_code(
                query,
                generated_code,
                state["error"],
                model=self.model
                )
            print(corrected_code)#debug
            state = self.code_executor.run(self.extract_code_from_generation(corrected_code))
            print(state)#debug
            return state["output"]


#not fully implemented 
#for conversations with memory 
class ChatAnalyst(Analyst):
    def __init__(self,df,client,model=None,code_executor:type[SimpleCodeExecutor]=SimpleCodeExecutor):
        super(ChatAnalyst).__init__(df=df,model=model,client=client,code_executor=code_executor)
        self.chat_transcript = []

    def chat(self, query):
        self.chat_transcript.append({"role":"user","content":query})
        self.chat_transcript.append({"role":"assistant","content":self.run_chain(
            self.chat_transcript[-1]["content"]
        )}) 
    
    def save_transcript(self,loc=os.getcwd()):
        json.dump(self.chat_transcript,fp=loc) 

