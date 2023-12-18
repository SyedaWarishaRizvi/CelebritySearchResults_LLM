import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory #storing conversation in memory - saved somewhere
import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

st.title('Celebrity Search')
input_text=st.text_input("Search the celebrity you want")

# Memory -- saved inside the LLM chain
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
description_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

# First Prompt template 
first_input_prompt=PromptTemplate(
    input_variables=['name'], # This is what you are searching for 
    template="Tell me about celebrity {name}"
)

# We have LLM chain to execute the prompt template 
llm=OpenAI(temperature=0.8)
chain1=LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key='person',memory=person_memory)

# Second Prompt template 
second_input_prompt=PromptTemplate(
    input_variables=['person'], # what you are searching for 
    template="When was {person} born"
)

chain2=LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob',memory=dob_memory)

# Third Prompt Template 
third_input_prompt=PromptTemplate(
    input_variables=['dob'], # what you are searching for 
    template="Mention 5 major events happened around {dob}"
)

chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='description',memory=description_memory)

#parent_chain=SimpleSequentialChain(chains=[chain1,chain2],verbose=True) #shows last result only - problem
parent_chain=SequentialChain(chains=[chain1,chain2,chain3],input_variables=['name'],output_variables=['person','dob','description'],verbose=True) # shows all results of all inputs

if input_text:
    st.write(parent_chain({'name':input_text}))
    
    with st.expander('Person Name'): 
        st.info(person_memory.buffer)

    with st.expander('Major Events'): 
        st.info(description_memory.buffer)