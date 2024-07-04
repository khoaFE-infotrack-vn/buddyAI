from fastapi import FastAPI
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings

from langchain_google_genai import GoogleGenerativeAIEmbeddings #to embed the text
# import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI #
from langchain.chains.question_answering import load_qa_chain #to chain the prompts
from langchain.prompts import PromptTemplate, MessagesPlaceholder #to create prompt templates

from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
from pydantic import BaseModel
import os
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

load_dotenv()

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

conversation_chain = None

def get_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001") # google embeddings
 
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index") # save the embeddings in local


# def get_conversation_chain(vectorstore):
#     llm = ChatOpenAI()
#     # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

#     memory = ConversationBufferMemory(
#         memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain

def get_conversation_chain():

    # define the prompt
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "Answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    
    Answer:
    """
    # model = ChatGoogleGenerativeAI(model = "gemini-pro", temperatue = 0.3) # create object of gemini-pro
    model = ChatOpenAI(
        model="google/gemini-flash-1.5:openrouter",
        temperature=0.1,
        max_tokens=2048,
        timeout=None,
        max_retries=2,
        api_key="itk_41590ef87a5b8eecbe0dd0a42fced9b9",
        base_url='https://stagesearch.infotrack.com.au/services/platform-ai-api/v1',
    )
    prompt = PromptTemplate(template = prompt_template, input_variables= ["context", "question"])
    # chat_history = ConversationBufferMemory(memory_key="chat_history")
    chain = load_qa_chain(model,chain_type="stuff", prompt = prompt)

    return chain

def format_chat_history(chat_history):
    formatted_history = "\n".join([f"Q: {entry['question']}\nA: {entry['answer']}" for entry in chat_history])
    return formatted_history


def user_input(user_question):
    # user_question is the input question
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # load the local faiss db
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization = True)

    # using similarity search, get the answer based on the input
    docs = new_db.similarity_search(user_question)

    chain = get_conversation_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question},
        return_only_outputs= True
        )

    print(response)
    return response
    # st.write("Reply: ", response["output_text"])

@app.on_event("startup")
async def startup_event():
    pdf_dir = "docspdf"  # Directory containing PDF files
    pdf_paths = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    pdf_text = get_pdf_text(pdf_paths)
    text_chunks = get_text_chunks(pdf_text)
    get_vectorstore(text_chunks)
    # vectorstore = get_vectorstore(text_chunks)
    # global conversation_chain
    # conversation_chain = get_conversation_chain(vectorstore)
    # conversation_chain = get_conversation_chain()

@app.post("/ask_question/")
async def ask_question(chat_request: ChatRequest):
    # global conversation_chain
    # if conversation_chain is None:
    #     return JSONResponse(status_code=400, content={"message": "No PDFs have been processed yet."})

    # response = conversation_chain({'question': chat_request.question})
    # return {"chat_history": response['chat_history']}
    response = user_input(chat_request.question)
    return {"chat_history": response['output_text']}

@app.get("/")
async def root():
    return {"message": "Welcome to the MultiPDF Chat API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
