def get_compression_retriever(vectordb, llm):
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    
    compressor = LLMChainExtractor.from_llm(llm)

    # create a compression retriever still using MMR as search type
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectordb.as_retriever(search_type="mmr")
    )
    return compression_retriever
