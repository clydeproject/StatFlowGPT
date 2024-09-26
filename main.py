from wrappers import OpenAIWrapper,TogetherAPIWrapper
from utils import Analyst

import pandas as pd 
import streamlit as st 

client = TogetherAPIWrapper()

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
                        model = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
                        agent = Analyst(df=df,client=client,model=model)
                        st.markdown("### Analysis Summary")
                        summary = agent.summarize_results(results=agent.run_chain(prompt),stream=True,model=model,query=prompt)
                        summary_placeholder = st.empty()
                        full_summary = ""
                        for chunk in summary:
                            full_summary += chunk["choices"][0]["text"]
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