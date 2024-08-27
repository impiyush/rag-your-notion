import os
import shutil
from langchain_chroma import Chroma

def load_from_notion(notion_db_path:str) -> list:
    from langchain_community.document_loaders import NotionDirectoryLoader
    loader = NotionDirectoryLoader(notion_db_path)
    docs = loader.load()
    if docs:
        # TODO: do more data cleaning
        # print(docs[0].page_content[0:200])
        # print(docs[0].metadata)
        return docs

    raise ValueError("Unable to load documents!")
    

def split_md_docs(docs:list) -> list:
    from langchain_text_splitters.markdown import MarkdownTextSplitter
    # NOTE: MarkdownTextSplitter is a wrapper for RecursiveTextSplitter for markdowns
    chunk_size = 500
    chunk_overlap = 50

    # MD splits
    markdown_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    md_splits = markdown_splitter.split_documents(docs)
    # print(md_splits[0])
    print(len(md_splits))
    # print(type(md_splits))
    return md_splits

def load_embedding_model():
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embed_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embed_model

def create_db(doc_splits:list, db_path:str):
    embed_model = load_embedding_model()

    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    vectordb = Chroma.from_documents(
        documents=doc_splits,
        embedding=embed_model,
        persist_directory=db_path
    )
    # print(vectordb._collection.count())
    vectordb.persist()
    return vectordb

def load_db(db_path:str):
    embed_model = load_embedding_model()
    return Chroma(persist_directory=db_path, embedding_function=embed_model)

    