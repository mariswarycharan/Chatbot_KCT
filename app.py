import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
# from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

st.cache_resource(show_spinner=False)
def load_model():
    load_dotenv()
    os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                             temperature=0 , convert_system_message_to_human=True)
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    return model,embeddings

model,embeddings = load_model()

def website_to_text(url):
    import bs4, requests
    response = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'})
# Get all text from site
    soup = bs4.BeautifulSoup(response.text,features="html.parser")
    # Prints all text that are within <div> with the class `texts`
    return " ".join([ i.text for i in soup.findAll({"div":{"class":"texts"}})])


def get_pdf_text(docs):
    
    text = ''
    for pdf in docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    global embeddings
    # vector_store = Chroma.from_texts(text_chunks, embedding = embeddings , persist_directory="chroma_db")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

st.cache_resource(show_spinner=False)
def get_conversational_chain():
    
    system_prompt = """
Your name is AI Bot and you should also act like a natural bot to answer open-ended questions. You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 

IMPORTANT INSTRUCTIONS:
- you should be more intractive ai chatbot and conversation must be engage with user, it should not be like boring
- Response should be professional and gentle, don't use offensive language.
- Respone should be structured , professional , Point by point wise , bold , italic , bullet point wise.
- if the user query is an open-ended question and then you should act like a normal conversation chatbot.
- you want to generate "related question" with help of below context.
- Remember all the context and chat history the user has provided and answer the question in natural language.
- you must want to answer the question  if it is question is somewhat related to the below context.

Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history} \n
you must want to answer all question , if it is question is somewhat related to the below context.
Context: {context} \n
Follow Up Input: {input} \n

Helpful Answer:
"""

    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )
    
    # prompt  = hub.pull("langchain-ai/retrieval-qa-chat")
    # new_db = Chroma(persist_directory="chroma_db",embedding_function=embeddings)
    new_db = FAISS.load_local("faiss_index", embeddings , allow_dangerous_deserialization=True)
    new_db = new_db.as_retriever(k=6)
    
    history_aware_retriever = create_history_aware_retriever(
    model, new_db, prompt
    )
    
    question_answer_chain = create_stuff_documents_chain(model, prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain )
    
    return rag_chain


chain = get_conversational_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.extend(
    [
        HumanMessage(content="hi there"),
        AIMessage(content="hi how can i help you?"),
    ]
    )

def user_input(user_question):
    
    global new_db,chain
    
    # docs = new_db.similarity_search(user_question ,k=5,)
    
    response = chain.invoke({"input": user_question, "context": "Your relevant context goes here"    , "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.extend(
    [
        HumanMessage(content=user_question),
        AIMessage(content=response["answer"]),
    ]
    )
    
   
    
    return response["answer"]

# Initialize chat history
if "messages_document" not in st.session_state:
    st.session_state.messages_document = []
    
# Display chat messages from history on app rerun
for message in st.session_state.messages_document:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def main():
    import time

    if prompt := st.chat_input("What is up?"):
        
        st.chat_message("user").markdown(prompt)
        st.session_state.messages_document.append({"role": "user", "content": prompt})
        
        with st.spinner('Wait for it...........'):  
            start_time = time.time()
            response = user_input(prompt)
            end_time = time.time()
            print(end_time - start_time)
            st.markdown(response)
            # st.subheader("References : ")
            
            # for doc in source_docs:
            #     st.write(doc)
            #     st.markdown("""
                            
            #                 """)
            
        st.session_state.messages_document.append({"role": "assistant", "content": response})
        
    # with st.sidebar:
    #     st.title("Menu:")
    #     pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        # website_url = st.text_input('Enter the website url : ')
        
        
        # if st.button("Submit & Process"):
        #     with st.spinner("Processing..."):
                

                # if website_url != '':
                #     raw_text = website_to_text(website_url)
                #     text_chunks = get_text_chunks(raw_text)
                #     get_vector_store(text_chunks)
                #     st.success("Done")
                
                # raw_text = get_pdf_text(pdf_docs)
                # text_chunks = get_text_chunks(raw_text)
                # get_vector_store(text_chunks)
                # st.success("Done")

if __name__ == "__main__":
    main()
    
