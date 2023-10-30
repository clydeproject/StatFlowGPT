from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from dotenv import load_dotenv
import pandas as pd 
import streamlit as st 
import openai 
import re 
import os 

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

#wrapper around chatCompletion endpoint 
def chat_completion(**kwargs):
    response = openai.ChatCompletion.create(
        model=kwargs["model"],
        temperature = kwargs["temperature"],
        max_tokens = kwargs["max_tokens"],
        stream=kwargs["stream"],
        messages = kwargs["messages"],
    )
    if kwargs["stream"]:    
        return response
    return response["choices"][0]["message"]["content"]

#wrapper around completion endpoint 
def completion(**kwargs):
    response = openai.Completion.create(
        model = kwargs["model"],
        temperature = kwargs["temperature"],
        prompt = kwargs["prompt"],
        stream = kwargs["stream"],
        max_tokens = kwargs["max_tokens"],
    )
    if kwargs["stream"]:
        return response
    return response["choices"][0]["text"]

def rephrase_query(**kwargs):
    '''
    df - pandas data frame,\n
    query - users query,\n
    '''
    system_prompt = f'''
    You must rephrase users queries about a dataframe into a more robust,specific and clearer/instructive manner for an 
    AI(GPT) code generation system that generates python code to answer the query.

    Here is some metadata about the dataframe.
    {kwargs["df"].head(5)}
    Here are all the names of the columns present in the dataframe:
    {kwargs["df"].columns}

    Here are some examples:
    1.
    users query: create a neural network to predict gender based on charges, test for (charges=15,000)
    Rephrased query: 
    Develop a binary classification neural network model with charges as the input feature and gender as the target variable. After training the model, apply it to predict the gender for a new data point with charges equal to $15,000."

    2.
    users query: logistic regression for age and gender
    Rephrased query: 
    Create a logistic regression model to predict gender based on age.

    3.
    users query: pie chart for current education level
    Rephrased query: 
    Generate a pie chart to visualize the distribution of current educational levels in the dataframe.

    =====================
    The user asked the following query: {kwargs["query"]}
    Rephrase it to aid better python code generation.

'''
    response = completion(model="gpt-3.5-turbo-instruct",temperature=0,prompt=system_prompt,max_tokens=400,stream=False)
    return response

def generate_code(**kwargs):
    '''
    object should have the following keys:
    query - the users query,
    df - dataframe object
    '''

    df = kwargs["df"]
    query = kwargs["query"]
    
    prompt = f'''
    You are given a pandas data frame with {df.shape[0]} rows and {df.shape[1]} columns.
    Here are the names of the columns: {df.columns}
    These are the first 5 rows of the data frame:
    {df.head(n=5)}.

    When asked about the data, your response should be python code that answers the users query .includes all the necessary libraries and describes the dataframe `df`.
    Using the provided dataframe, df , return python code only. 
    here is how the starting of the code should look like.
    
```python
import pandas as pd
#you can import other modules if necessary
    
df = pd.DataFrame(df)


```
Here are some examples

1. 
query - fit a linear regression on Age and Monthly expenses 
code generated:
```python
import pandas as pd
import statsmodels.api as sm

df = df

# Fit a linear regression model on age and monthly expenses
X = df['Age']
y = df['Final Monthly Expense']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Print out the regression results
print(model.summary())
```

2. 
query - show me the distribution of Final monthly expenses 
code generated:
```python
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

df = df

# create a bar chart of final monthly expenses
fig, ax = plt.subplots()
ax.bar(df['Name'], df['Final Monthly Expense'])
ax.set_xlabel('Name')
ax.set_ylabel('Final Monthly Expense')
ax.set_title('Final Monthly Expenses for 200 Students')
st.pyplot(fig)
```

3. 
query - perform a chisquare goodness of fit test for age
code generated:
```python
import pandas as pd
import scipy.stats as stats
import streamlit as st
import matplotlib.pyplot as plt

df = df

# perform chi-square goodness of fit test for age
observed = df['Age'].value_counts()
expected = df['Age'].value_counts().mean()
chi2, p = stats.chisquare(observed, expected)

# display results
print("Chi-square statistic:", chi2)
print("p-value:", p)

#create bar chart of age distribution
fig, ax = plt.subplots()
ax.bar(observed.index, observed.values)
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.set_title("Age Distribution")
st.pyplot(fig)
```

4. 
query - show me the graph of open and give me statistics
code generated:
```python
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

df = df

# calculate statistics for open prices
print("Mean:", df['open'].mean())
print("Median:", df['open'].median())
print("Standard Deviation:", df['open'].std())

# create a line graph of open prices
fig, ax = plt.subplots()
ax.plot(df['date'], df['open'])
ax.set_xlabel('Date')
ax.set_ylabel('Open Price')
ax.set_title('Open Prices for TSLA')
st.pyplot(fig)

```

5.
query - check if percent change follows normal distribution and show me a graph
code generated:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import streamlit as st

df = df

# Calculate percent change
df['percent_change'] = df['change_percent'].pct_change()

# Remove NaN values
df = df.dropna(subset=['percent_change'])

# Check if percent change follows normal distribution
k2, p = stats.normaltest(df['percent_change'])
alpha = 0.05
if p < alpha:
    print("The percent change does not follow a normal distribution.")
    print("p value ",p)
    print("alpha value ",alpha)
else:
    print("The percent change follows a normal distribution.")
    print("p value :",p)
    print("alpha value :",alpha)

# Plot histogram of percent change
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['percent_change'], kde=True, ax=ax)
plt.xlabel('Percent Change')
plt.ylabel('Frequency')
plt.title('Histogram of Percent Change')
st.pyplot(fig)
```
'''
    
    if kwargs["model"] == "gpt-3.5-instruct":
        prompt+= f'''

IMPORTANT - ALWAYS PRINT OUT FINAL RESULTS
now generate code based on the examples for this query
QUERY : {query}
'''
        response = completion(model="gpt-3.5-turbo-instruct",temperature=0,prompt=prompt,max_tokens=1000,stream=False)

        #Generated code from LLM
        generated_code =  response
        print(f"QUERY: {query}")#debug
        print(f"GENERATED CODE:\n{generated_code}")#debug
        return extract_code(generated_code)
        
    #for better code generation(slower)
    elif kwargs["model"] == "gpt-4":
        response = chat_completion(
            model=kwargs["model"],
            temperature=0,
            messages= [{"role":"system","content":prompt},
                        {"role":"user","content":query},],
            max_tokens = 1000,
            stream=False,
        )
        generated_code = response
        print(f"QUERY: {query}")#debug
        print(f"GENERATED CODE:\n{generated_code}")#debug
        return extract_code(generated_code)
    

def extract_code(text):
    code_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
    python_code_blocks = [block.strip()[7:] for block in code_blocks if block.startswith('python')]
    if python_code_blocks:
        return '\n'.join(python_code_blocks)
    # If no code blocks with ```python``` delimiters are found, assume the entire input text is the code.
    return text

def error_correct_code(code,error_message,query):

    system_prompt = f'''
you are an expert at error correcting python code, you are given a pandas data frame with 
{df.shape[0]} rows and {df.shape[1]} columns. 
Here is the metadata of the data frame:
{df.head(n=5)}.

when asked to correct code you should return the corrected code.
here is 1 example :

1. 
query -  perform a logistic regression which predicts sex based on charges
code with error - 
```python
import pandas as pd
import statsmodels.api as sm

df = df

X = df['charges']
y = df['sex']
X = sm.add_constant(X)
model = sm.Logit(y, X).fit()

# Print out the regression results
print(model.summary())
```
Error message - 'raise ValueError("Pandas data cast to numpy dtype of object. "
                ValueError: Pandas data cast to numpy dtype of object. Check input data with np.asarray(data).'

Corrected Code - 
```python
import pandas as pd
import statsmodels.api as sm

df = df

# Create a binary 'sex' column (1 for 'female', 0 for 'male')
df['sex_binary'] = df['sex'].apply(lambda x: 1 if x == 'female' else 0)

# Fit a logistic regression model
X = df[['charges']]
y = df['sex_binary']
X = sm.add_constant(X)
model = sm.Logit(y, X).fit()

# Print out the regression results
print(model.summary())
```
=============================================
Correct this code now

query - {query}

code with error - 
{code}

error message - {error_message}

'''
    response = completion(model="gpt-3.5-turbo-instruct",temperature=0,prompt=system_prompt,max_tokens=1000,stream=False)
    corrected_code = response
    return extract_code(corrected_code)

def execute_code(code, query):
    output_buffer = StringIO()
    output = None
    error = None
    fix_count = 1
    code_state = code

    while fix_count <= 2:
        with redirect_stdout(output_buffer):
            with redirect_stderr(output_buffer):
                try:
                    exec(code_state)
                    output = output_buffer.getvalue()
                    return output
                except Exception as e:
                    error = str(e)
                    print(f"ERROR IN CODE : {error}")#debug
                    #print(error)
                    code_state = error_correct_code(code_state, error_message=error, query=query)
                    print(f"CORRECTED CODE({fix_count}): \n",code_state)#debug
                    fix_count += 1

    return output  # Return the final captured output after 5 attempts

def chat(prompt,description):
    global df
    user_query = prompt
    query_response = execute_code(generate_code(df=df,query=user_query,model="gpt-3.5-instruct"),query=user_query)
    print(f"OUTPUT:\n{query_response}\n")#Debug
    print("================================================\n\n")

    system_prompt = f'''
You are a highly intelligent data chatbot who is a data scientist with access to a user-provided data set.
 Your mission is to assist users in exploring, analyzing, and gaining insights from 
 the data they provide. Engage in a conversation with users, understand their queries, 
 and utilize the data set to provide accurate and informative responses. Your goal is to be a 
 reliable data analysis companion, helping users make informed decisions and discover meaningful 
 patterns in their data.

 here is the meta data of the data frame the user is querying:
{df.head(5)}

 Here is a description of the dataframe:
 {description}

Your responses should be of the following manner, here are some examples
1. The mean is.. # some number
2. Heres the graph of ....

Answer the following question about the data frame with the help of the provided context, be very precise and dont leave any important/relevant details out, also
dont say/mention anything that isnt present in the context.
context:
{query_response}

question:{user_query}
'''
    #result summarization model can be different 
    response = completion(model="gpt-3.5-turbo-instruct",temperature=0.1,prompt=system_prompt,max_tokens=1000,stream=True)
    return response


if __name__ == "__main__":
    # streamlit page config
    st.set_page_config(
    page_title="Data Chat",
    layout="wide",
    )         
    st.header("StatFlowGPT🎓🪄")
    #page elements [ tabs]
    chat_area,data_area = st.tabs(["Chat","View/Configure Data"])
    output_text = st.container()
    
    with data_area:
        
        uploaded_file = st.file_uploader(label="Upload a data set(.CSV)")
        #uploaded_file_path = st.text_input(placeholder="Enter the path of the data set(.CSV)",label=" ")
        description = st.text_input(placeholder="Please provide additional information about the data set, ex- column context etc",label=" ")
            
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(data=df,use_container_width=True,)
        
        with chat_area:
            prompt_ = st.text_input(placeholder="Enter a prompt here",label=" ",)
            if prompt_:
                prompt_ = rephrase_query(query=prompt_,df=df)#rephrases it 
                output = chat(prompt=prompt_,description=description)
                text=st.empty()
                answer = ""
                #streaming responses 
                for chunk in output:
                    event_text = chunk["choices"][0]["text"]
                    answer+= event_text
                    result = "".join(answer).strip()
                    result.replace("\n","")
                    text.markdown(body=result)
    
    #Hides streamlit watermark
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
