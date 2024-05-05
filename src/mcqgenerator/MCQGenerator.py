import os 
import re
import json 
import pandas as pd
import traceback
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging 
import langchain
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain, SequentialChain, ConversationChain
from langchain_community.callbacks import get_openai_callback
# from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentType
from langchain.agents import load_tools 
from langchain.agents import initialize_agent
import PyPDF2



# Load environment variables from .env file
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY") #openAI API
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") # Hugging Face API
serpapi_key = os.getenv("serpapi_key") # Search-result API
os.environ['HUGGINGFACEHUB_API_TOKEN']=HUGGINGFACEHUB_API_TOKEN 


# LLM model from hugging face
repo_id =  "mistralai/Mistral-7B-Instruct-v0.2" 

llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=200, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN
)

# Template
TEMPLATE = """
Text:{text}
You are an expert MCQ Marker. Given the above text, it is your job to create a Quiz of {number} multiple choice questions
for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide.
Ensure to make {number} MCQs.
### RESPONSE_JSON 
{response_json}

"""
# Input Prompt template
quiz_gen_prompt = PromptTemplate(
    input_variables = ["text", "number", "subject","tone","response_json"],
    template = TEMPLATE
)

# Connect llm with prompt via llmchain
quiz_chain = LLMChain(prompt=quiz_gen_prompt, llm=llm,output_key='quiz',verbose=True)

TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_eval_prompt=PromptTemplate(
    input_variables=["subject", "quiz"],
    template=TEMPLATE2
)

review_chain=LLMChain(llm=llm, prompt=quiz_eval_prompt, output_key="review", verbose=True)

## Sequential chain 
gen_eval_chain=SequentialChain(chains=[quiz_chain, review_chain], input_variables=["text", "number", "subject", "tone", "response_json"],
                                        output_variables=["quiz", "review"], verbose=True,)
