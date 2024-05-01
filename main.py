from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
import uvicorn
from pdf_qa import PdfQA
from constants import *
import os


app = FastAPI()

bot = PdfQA()
emb = EMB_OPENAI_ADA
llm = LLM_OPENAI_GPT35
load_in_8bit = False


class Question(BaseModel):
    question: str 
    
    
def load_llm(llm,load_in_8bit):
    if llm == LLM_OPENAI_GPT35:
        return bot.create_openai_35(load_in_8bit)
    else:
        raise ValueError("Invalid LLM setting")

def load_emb(emb):
    if emb == EMB_OPENAI_ADA:
        return bot.create_openai_embaddings()
    else:
        raise ValueError("Invalid embedding setting")
    
@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}


@app.post('/api/train_model')
async def trainModel():
    try:

        bot.config = {
                "pdf_path": str('B1'),
                "embedding": emb,
                "llm": llm,
                "load_in_8bit": False
        }
        bot.embedding = load_emb(emb)
        bot.llm = load_llm(llm,False)
        bot.init_embeddings()
        bot.init_models()

        bot.vector_db_pdf()
        return {"msg":"Model Trained Successfully!!!"}
    
    except:
        return {"msg":"Internal Server Error!!!"}
        
@app.get('/api/ask')
async def askQuery(question:Question):
    try:
        bot = PdfQA() # Remove this for chat history
        bot.config = {
            "pdf_path": str('B1'),
            "embedding": emb,
            "llm": llm,
            "load_in_8bit": load_in_8bit
        }
        bot.embedding = load_emb(emb)
        bot.llm = load_llm(llm,load_in_8bit)        
        bot.init_embeddings()
        bot.init_models()
        bot.retreival_qa_chain()
        # question = question + " Provide 5-7 key points on it with explaination and example."
        answer = bot.answer_query(question.question)
        
        return {"msg":"Answer Generated Successfully!!!","question":question.question,"answer":answer}
    except:
        return {"msg":"Internal Server Error!!!"}
    
    
    
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Wait for a message from the client
        #data = await websocket.receive_text()
        data = await websocket.receive_json()
        data = data["question"]
        
        # Perform calculations based on the received data
        result = generate_answer(data)
        
        # Send the result back to the client
        #await websocket.send_text(result)
        response = {"question":data,"answer":result}
        
        await websocket.send_json(response)


def generate_answer(data:str) -> str:

    
    bot = PdfQA() # Remove this for chat history
    bot.config = {
        "pdf_path": str('B1'),
        "embedding": emb,
        "llm": llm,
        "load_in_8bit": load_in_8bit
    }
    bot.embedding = load_emb(emb)
    bot.llm = load_llm(llm,load_in_8bit)        
    bot.init_embeddings()
    bot.init_models()
    bot.retreival_qa_chain()
    # question = question + " Provide 5-7 key points on it with explaination and example."
    answer = bot.answer_query(data)
    
    return answer
    





































































































































































































































































































































































































































































































































































































































































