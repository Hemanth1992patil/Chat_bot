import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import os
from flask import Flask, jsonify,request

app = Flask(__name__)

# Define a route for the root URL ("/") using the route() decorator
@app.route('/search', methods = ['POST'])
def openAi_api():    
    search_word = request.json.get("question")
    print(search_word)
    # os.environ["OPENAI_API_KEY"] = "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"
    os.environ["OPENAI_API_KEY"] = "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"
    
    data = pd.read_csv('output.csv')
    data_df = pd.DataFrame(data)
    pd_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), data_df, verbose=True)
    return pd_agent.run(search_word)
if __name__ == '__main__':
    app.run(debug=True) 
    
