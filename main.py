from wrappers import OpenAIWrapper,TogetherAPIWrapper,LiteLLMWrapper
from utils_ import Analyst

import pandas as pd 
import streamlit as st 

#can use any model/custom finetuned model for each task,
#IMPORTANT: CODE GENERATION AND ERROR CORRECTION FOR ANY MODEL IS PROMPT SPECIFIC, THE PROMPTS IN USE(prompt_templates.py) is written for meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
#if changing model,modify the underlying prompts for better performance

model_stack = {
    "rephrase_query":"together_ai/mistralai/Mistral-7B-Instruct-v0.2",
    "code_generation":"together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "error_correction":"together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    }

client = LiteLLMWrapper()


custom_css = """
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTextInput > div > div > input {
        background-color: #f8f9fa;
    }
    .stTextArea > div > div > textarea {
        background-color: #f8f9fa;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .css-1v0mbdj.etr89bj1 {
        border: none;
        box-shadow: none;
    }
</style>
"""

def main():
    st.set_page_config(page_title="StatFlowGPT", layout="wide")
    st.markdown(custom_css, unsafe_allow_html=True)

    st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>StatFlowGPT 🎓🪄</h1>
    <p style='text-align: center; font-style: italic;'>Chat with your data</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Data Configuration")
        uploaded_file = st.file_uploader("Upload a data set (.CSV)", type="csv")

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("Data uploaded successfully!")
            with st.expander("View Data Preview"):
                st.dataframe(df.head())
        else:
            df = None
            st.warning("Please upload a CSV file to start analyzing.")

    with col2:
        st.markdown("### Data Chat")
        if df is not None:
            prompt = st.text_input("Ask a question about your data", key="prompt")
            if st.button("Analyze", key="analyze_button"):
                if prompt:
                    with st.spinner("Analyzing your data..."):
                        agent = Analyst(df=df,client=client,model_stack=model_stack)
                        st.markdown("### Analysis Summary")
                        summary = agent.summarize_results(results=agent.run_chain(prompt),stream=True,model="together_ai/mistralai/Mistral-7B-Instruct-v0.2",query=prompt)
                        summary_placeholder = st.empty()
                        full_summary = ""
                        for chunk in summary:
                            content = chunk.choices[0].delta.content
                            if content is not None:#to handel stream of a None type chunk at the end of a completion 
                                full_summary += chunk.choices[0].delta.content
                            summary_placeholder.markdown(full_summary)
                else:
                    st.warning("Please enter a question to analyze your data.")
        else:
            st.info("Upload your data in the left panel to start chatting about it!")

    # Footer
    st.markdown("""
    <div style='text-align: center; margin-top: 2rem;'>
        <p>Prototype created by Paartha Nimbalkar</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
