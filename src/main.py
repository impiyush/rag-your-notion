from load_docs import load_from_notion, split_md_docs, create_db, load_db
import os
from langchain_ollama.llms import OllamaLLM
from retriever import get_compression_retriever
from chat import retrievalqa_chain

llm = OllamaLLM(model="llama3", temperature=0)

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

if __name__ == "__main__":
    notion_docs = load_from_notion("../../SecondBrain_NotionDB")
    notion_docs_splits = split_md_docs(notion_docs)
    vectordb_dir = "./db/chroma/"

    if os.path.exists(vectordb_dir):
        vectordb=load_db(vectordb_dir)
    else:
        vectordb = create_db(notion_docs_splits)

    # ----- retrieval -----
    question = "what is the name of my manager at NIKE?"
    # docs = vectordb.similarity_search(question,k=3)
    # print(docs)

    # docs_mmr = vectordb.max_marginal_relevance_search(question,k=3)
    # print("\n---------------\n")
    # print(docs_mmr)

    # compression retrieval
    compression_retriever = get_compression_retriever(vectordb, llm)
    # compressed_docs = compression_retriever.get_relevant_documents(question)
    # pretty_print_docs(compressed_docs)
    
    # ----- Chat -----
    retrievalqa_chain(question, compression_retriever)
