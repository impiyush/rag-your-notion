from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

def retrievalqa_chain(question, retriever):
    # Prompt
    template = """You are an agent helping prepare for interviews. 
    Answer the question using only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Local LLM
    llm = ChatOllama(model="llama3", temperature=0)

    # Chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    chain.invoke(question)