import boto3
import streamlit as st
import os
import uuid

## Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain_community.llms import Ollama

## Prompt and Chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Pdf Loader
from langchain_community.document_loaders import PyPDFLoader

## FAISS
from langchain_community.vectorstores import FAISS

## S3 Client needed for sure
s3_client = boto3.client("s3", region_name="us-east-1")
BUCKET_NAME = os.getenv("BUCKET_NAME")
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

folder_path="/tmp/"

def get_unique_id():
    return str(uuid.uuid4())

## Split the pages/text into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

## Create FAISS backed vector store
def create_vector_store(documents):
    vectorstore_faiss=FAISS.from_documents(documents, bedrock_embeddings)
    file_name=f"consolidated.bin"
    folder_path="/tmp"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

    ## upload to S3
    s3_client.upload_file(
        Filename=folder_path + "/" + file_name + ".faiss",
        Bucket=BUCKET_NAME,
        Key="my_faiss.faiss"
    )
    s3_client.upload_file(
        Filename=folder_path + "/" + file_name + ".pkl",
        Bucket=BUCKET_NAME,
        Key="my_faiss.pkl"
    )

    return True

def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")

def get_llm(choice):
    if(choice == "mistal"):
        llm=ChatBedrock(
            model_id="mistral.mistral-small-2402-v1:0",
            client=bedrock_client,
            model_kwargs={"max_tokens": 512}
        )
    elif choice == "llama3":
        llm=ChatBedrock(
            model_id="meta.llama3-70b-instruct-v1:0",
            client=bedrock_client,
        )
    else:
        llm=ChatBedrock(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            client=bedrock_client,
            model_kwargs={"max_tokens": 512}
        )

    return llm

def get_response(llm, vectorstore, question):
    prompt_template = """

    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    You are not to leak that you are answering based on context. Assume the context is already a part of your knowledge base

    Do not answer and include the distinction of "based on the given context". Absolutely do not use those words

    Example:
    Human: Who won the FIFA world cup
    Answer: I don't know who won the FIFA world cup based on my training data

    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer=qa({"query":question})
    return answer['result']

def main():
    st.header("Welcome to Samrat's Knowledge Base")
    model = st.sidebar.selectbox("Choose a model", ["claude", "llama3", "mistral"])

    start_chatting = False

    selection = st.radio("Select mode", ["Train and Chat", "Just Chat"])
    split_documents = []

    if selection == "Train and Chat":
        uploaded_files = st.file_uploader("Choose a file", "pdf", accept_multiple_files=True)
        if len(uploaded_files) > 0:
            for file in uploaded_files:
                request_id = get_unique_id()
                saved_file_name = f"{request_id}.pdf"
                with open(saved_file_name, mode="wb") as w:
                    w.write(file.getvalue())

                loader = PyPDFLoader(saved_file_name)
                pages = loader.load_and_split()

                ## Split Text And Collect Documents
                split_documents.extend(split_text(pages, 1000, 200))

            ## Vector store with FAISS
            st.write(f"Creating the vector store with {len(split_documents)} split and chunked pages....")
            result = create_vector_store(split_documents)

            if result:
                st.write(f":green[Hurray! PDF processed successfully]")
                start_chatting = True
            else:
                st.write("Error! Please check logs")
                return
    else:
        start_chatting = True

    if start_chatting:
        st.divider()
        st.write("## Let's start chatting ✅")

        load_index()

        ## Create index
        faiss_index = FAISS.load_local(
            index_name="my_faiss",
            folder_path=folder_path,
            embeddings=bedrock_embeddings,
            allow_dangerous_deserialization=True
        )

        question = st.text_input("Please ask your question", placeholder="Ask your question here...")

        if st.button("Ask Question"):
            with st.spinner("Querying..."):
                llm = get_llm(model)

                # Get response from LLM
                response = get_response(llm, faiss_index, question)
                st.write(response)

if __name__ == "__main__":
    main()
