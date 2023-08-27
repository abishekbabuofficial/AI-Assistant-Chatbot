from langchain import HuggingFaceHub, PromptTemplate, LLMChain
import chainlit as cl
import os
import gradio as gr

HF_API_TOKEN = "Enter you HF API Token here"
os.environ['HF_API_TOKEN'] = HF_API_TOKEN

model_id = "tiiuae/falcon-7b-instruct"
LLM = HuggingFaceHub(huggingfacehub_api_token=HF_API_TOKEN, 
                     repo_id=model_id,
                     model_kwargs={"temperature":0.7,"max_new_tokens":700})

from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=3)

template = """
You are Abi, a useful AI assistant and answer for user questions.
Question: {question}
Answer:
"""

promt = PromptTemplate(template = template, input_variables=['question'])
llm_chain = LLMChain(llm=LLM, prompt=promt, memory= memory)

def ask(question,history):
  ans = llm_chain.run(question)
  return ans

demo = gr.ChatInterface(ask,
                        examples=['How are you doing?','What can you do?','Tell me a story','Tell me a joke','Who is prime minister of India?'],
                        title="Abi - An AI Assistant",
                        description="This is an AI Chatbot Assistant that uses Falcon 7b-instruct LLM to generate the output for the user queries.\n You can further know more from my GitHub Repo👉 https://github.com/abishekbabuofficial/AI-Assistant-Chatbot.\n♥️♥️♥️Application developed by Abishek B♥️♥️♥️")

demo.launch(debug=True)
