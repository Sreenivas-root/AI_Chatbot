import chatbot_usecase as usecase
import chat_history_usecase as history_usecase
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def run_inference(query, session_id):
    vectordb = usecase.create_or_load_embeddings_db()
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    llm = usecase.llm_fn()
    # qa_chain = usecase.get_response(llm, retriever)
    # llm_response = qa_chain.invoke(query)
    # return usecase.wrap_text_preserve_newlines(llm_response['result'])
    doc_chain = history_usecase.get_history_aware_response(llm, retriever)
    full_chain = RunnableWithMessageHistory(
        doc_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    c = 0
    for chunk in full_chain.stream(
        {"input": query},
        config={
            "configurable": {"session_id": session_id}
        },  # constructs a key "abc123" in `store`.
    ):
        print('Chunk %d' % c)
        c+=1
        print(chunk)
        if 'answer' in chunk:
            yield str(chunk['answer'])

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
