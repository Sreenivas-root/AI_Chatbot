import chatbot_usecase as usecase

def run_inference(query):
    vectordb = usecase.create_or_load_embeddings_db()
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    llm = usecase.llm_fn()
    qa_chain = usecase.get_response(llm, retriever)
    llm_response = qa_chain.invoke(query)
    # return llm_response
    return usecase.wrap_text_preserve_newlines(llm_response['result'])