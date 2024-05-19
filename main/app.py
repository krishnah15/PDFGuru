import os
from typing import List
from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyMuPDFLoader
from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.embeddings import OllamaEmbeddings
import chainlit as cl
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema.runnable.config import RunnableConfig


@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload your PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
    await msg.send()

    loader = PyMuPDFLoader(file.path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    # Create a Chroma vector store
    vector_db = await cl.make_async(Chroma.from_documents)(
        documents=chunks, 
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        collection_name="newnew"
    )

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        ChatOllama(model="llama2"),
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        memory=memory,
        return_source_documents=True, 
    ) 

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb=cl.AsyncLangchainCallbackHandler()
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()