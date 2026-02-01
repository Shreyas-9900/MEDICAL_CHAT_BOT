        # from flask import Flask, render_template, request
        # from src.helper import download_hugging_face_embeddings
        # from langchain_pinecone import PineconeVectorStore
        # from langchain.chains.retrieval import create_retrieval_chain
        # from langchain.chains.combine_documents import create_stuff_documents_chain
        # from langchain_core.prompts import ChatPromptTemplate
        # from dotenv import load_dotenv
        # import os

        # app = Flask(__name__)

        # load_dotenv()

        # PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

        # embeddings = download_hugging_face_embeddings()

        # index_name = "medicalbot"

        # docsearch = PineconeVectorStore.from_existing_index(
        #     index_name=index_name,
        #     embedding=embeddings,
        # )

        # retriever = docsearch.as_retriever(search_kwargs={"k": 3})

        # llm = ChatOpenAI(model="gpt-3.5-turbo")

        # system_prompt = "You are a helpful medical assistant chatbot."

        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", system_prompt),
        #         ("human", 
        #          "Use the following context to answer the question:\n\n"
        #          "{context}\n\n"
        #          "Question: {input}")
        #     ]
        # )


        # question_answer_chain = create_stuff_documents_chain(llm, prompt)

        # rag_chain = create_retrieval_chain(retriever, question_answer_chain)


        # @app.route("/")
        # def home():
        #     return render_template("home.html")



        # @app.route("/get", methods=["POST"])
        # def chat():
        #     msg = request.form["msg"]

        #     response = rag_chain.invoke({"input": msg})

        #     return str(response["answer"])


        # if __name__ == "__main__":
        #     app.run(host="0.0.0.0", port=8080, debug=True)


from flask import Flask, request
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.prompt import *
from flask import Flask, render_template, request, jsonify

# Flask App
app = Flask(__name__)

# Load ENV
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load Embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone Index
index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_kwargs={"k": 3})

# ✅ OFFLINE HuggingFace LLM
hf_pipeline = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=150
)



llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Context: {context}\n\nQuestion: {input}")
    ]
)

# RAG Chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# Home Route
from flask import render_template

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")

    response = rag_chain.invoke({"input": msg})

    return jsonify({"answer": response["answer"]})

# Run Server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
