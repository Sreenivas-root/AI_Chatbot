from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

# from InstructorEmbedding import INSTRUCTOR
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import secret

from langchain_core.prompts import PromptTemplate
from langchain.schema.prompt_template import format_document
import textwrap


def load_document_and_split():
    loader = PyPDFLoader('./docs/competitor_data.pdf')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

def create_or_load_embeddings_db(load=True):
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", cache_folder='model')
    persist_directory = 'db'
    embedding = instructor_embeddings
    if not load:
        texts = load_document_and_split()
        vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)
    vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)
    return vectordb

def llm_fn():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",
                 temperature=0.7, top_p=0.85, google_api_key=secret.GOOGLE_API_KEY)
    return llm

def get_prompt_template():
    # doc_prompt = PromptTemplate.from_template("{page_content}")
    llm_prompt_template = """Answer the question based on the following context:
    {context}
    If you don't have the context or the context does not provide any useful information, write based on what you think might be the answer for this question.
    Question: {question}
    """
    llm_prompt = PromptTemplate.from_template(llm_prompt_template)
    return llm_prompt

def get_response(llm, retriever):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)
    return qa_chain

def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata)