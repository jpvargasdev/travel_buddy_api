import os
from langchain.base_language import BaseLanguageModel

from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import CSVLoader 

EXAMPLE_CSV = os.getcwd() + "/app/agent/example.csv"

def charge_csv(
        llm: BaseLanguageModel,
        path: str = EXAMPLE_CSV
        ) -> BaseRetrievalQA: 
    loader = CSVLoader(path)
    documents = loader.load_and_split()
    embeddings = OpenAIEmbeddings(client="")
    doc_search = Chroma.from_documents(documents, embeddings)
    qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=doc_search.as_retriever(),
            )
    return qa

