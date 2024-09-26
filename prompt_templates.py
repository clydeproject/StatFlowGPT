REPHRASE_PROMPT = """Given a user's query about a dataframe, rephrase it into a more specific, clear, and actionable format suitable for AI-based Python code generation. Consider the following dataframe information:

Dataframe preview (first 5 rows):
{dfpreview}

Column names:
{colnames}

Original user query:
{userquery}

Please rephrase the query to:
1. Be more specific about the desired operations
2. Reference exact column names where applicable
3. Clarify any ambiguities in the original query
4. Include any relevant data types or constraints
5. Suggest appropriate Python libraries or functions if applicable
6. Only return the rephrased query"""


GENERATE_CODE = """Generate efficient and robust Python code to address the following query about a pandas DataFrame. Use the provided DataFrame information and structure:

DataFrame Summary:
- Rows: {nrows}
- Columns: {ncols}
- Column Names: {colnames}

DataFrame Preview (first 5 rows):
{dfhead}

Instructions:
1. Use the DataFrame named 'df' in your code.
2. Include necessary imports and data preprocessing steps.
3. Implement error handling and input validation where appropriate.
4. Use pandas and numpy operations for efficiency when possible.
5. Add brief comments to explain key steps or complex operations.
6. If applicable, include code to display or return the results.
7. Always print out final results

Provide only the Python code, starting with:

```python
import pandas as pd
# Import other necessary modules

df = df

# Your code here
```
Query:
{query}
"""

ERROR_CORRECT = """Analyze and correct the following Python code that encountered an error while processing. Please provide the corrected version of the code.

Query:
{query}

Code with Error:
```python
{error_code}

Error Message:
{error_message}

DataFrame Information:

Rows: {nrows}
Columns: {ncols}
Column Names: {colnames}
Data Preview (first 5 rows):
{dfhead}

Instructions:

Identify the root cause of the error.
Provide only the corrected version of the code that addresses the error.

Corrected Code:
```
# Imports
import pandas as pd

# Corrected code here```"""


SUMMARIZE_RESULTS = """As an intelligent data analysis assistant, provide a concise yet comprehensive summary of the query results based on the following information:
Question:
{query}

Answer:
{results}

Instructions for summarization:
1. Analyze the query results in the context of the original question.
2. Focus on key findings and insights directly related to the user's query.
3. Use specific data points or statistics from the results to support your summary.
4. Highlight any notable trends, patterns, or anomalies in the data.
5. If applicable, mention any limitations or caveats in the analysis.
6. Avoid introducing information not present in the provided context or data.
7. Use clear, concise language appropriate for both technical and non-technical audiences.
8. If relevant, suggest potential next steps or areas for further investigation.
9. Answer in a way that directly responds to the question

Summary:
[Provide your summary here, structured in clear paragraphs or bullet points as appropriate]

Key Takeaways (if applicable):
- [List 2-3 main points or conclusions]

Data Insights (if applicable):
- [Provide 1-2 specific data-driven insights]

Potential Next Steps (if applicable):
- [Suggest 1-2 follow-up actions or analyses]
"""