# from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import TokenTextSplitter
#from transformers import pipeline
# from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
#from langchain import HuggingFacePipeline
# from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.llms import OpenAI
from constants import *
# from transformers import AutoTokenizer
# import torch
import os
# import re
# from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS


class PdfQA:
    def __init__(self,config:dict = {}):
        self.config = config
        self.embedding = None
        self.vectordb = None
        self.llm = None
        self.qa = None
        self.retriever = None
        self.chat_history = []

    
    @classmethod
    def create_openai_embaddings(cls):
        embaddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        return embaddings
    
    @classmethod
    def create_openai_35(cls,load_in_8bit=False):
        llm = ChatOpenAI(model_name=LLM_OPENAI_GPT35, temperature=0.2,max_tokens=350)
        return llm
    
    
        
    def init_embeddings(self) -> None:
        # OpenAI ada embeddings API
        if self.config["embedding"] == EMB_OPENAI_ADA:
            self.embedding = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
            
        else:
            self.embedding = None ## DuckDb uses sbert embeddings
           
    def init_models(self) -> None:
        """ Initialize LLM models based on config """
        load_in_8bit = self.config.get("load_in_8bit",False)
        # OpenAI GPT 3.5 API
        if self.config["llm"] == LLM_OPENAI_GPT35:
            if self.llm is None:
                self.llm = PdfQA.create_openai_35(load_in_8bit=load_in_8bit)
        
        
        else:
            raise ValueError("Invalid config")        
    
    
    def vector_db_pdf(self,pdf_docs) -> None:
        """
        creates vector db for the embeddings and persists them or loads a vector db from the persist directory
        """
        persist_directory = self.config.get("persist_directory",None)

        text=""
        for pdf in pdf_docs:
            pdf_reader= PdfReader(pdf)
            for page in pdf_reader.pages:
                text+= page.extract_text()
            
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100, encoding_name="cl100k_base")  # This the encoding for text-embedding-ada-002
        texts = text_splitter.split_text(text)

            # self.vectordb = Chroma.from_texts(texts=texts, embedding=self.embedding, persist_directory=persist_directory)
        self.vectordb = FAISS.from_texts(texts=texts, embedding=self.embedding)
        self.vectordb.save_local("vector_db")
        

    def retreival_qa_chain(self):
        """
        Creates retrieval qa chain using vectordb as retrivar and LLM to complete the prompt
        """
        
        
        
        # if self.config["llm"] == LLM_OPENAI_GPT35:
          
        #   self.qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name=LLM_OPENAI_GPT35, temperature=0.2,max_tokens=1024),
        #                               self.vectordb.as_retriever(search_kwargs={"k":5}))
            
        vectorstore = FAISS.load_local("vector_db", self.embedding ,allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(model_name=LLM_OPENAI_GPT35, temperature=0.2,max_tokens=1024)
        self.qa = ConversationalRetrievalChain.from_llm(llm = llm, retriever = retriever)
          
          
          #self.qa.return_source_documents = True
            
    def answer_query(self,question:str) ->str:

        answer_dict = self.qa({"question":question,"chat_history":self.chat_history})
        print(answer_dict)
        answer = answer_dict["answer"]
        self.chat_history.append((question,answer)) # add query and response to the chat history
        
        
        return answer
    
    

