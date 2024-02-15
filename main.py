# Importación de Las librerias

from PyPDF2 import PdfReader #Librearia para manejar PDF
from langchain.text_splitter import RecursiveCharacterTextSplitter #Libreria para Splitear la informacion
from langchain.embeddings import HuggingFaceEmbeddings #Librearia para crear los embeddings
from langchain.vectorstores import FAISS #libreria para crear la data
from langchain.chains.question_answering import load_qa_chain #Libreria para cargar el modelo LLM
from langchain import HuggingFaceHub  #Libreraia para usar los modelos de hugginface
import os #libreria para usar el token de hugginFace
import streamlit as st #Necesaria para poder realizar la interfaz local


#Coloca el titulo de la pagina
st.title('Preguntar por a tu documento PDF')
#Necesario para subir el archivo
pdf_obj= st.file_uploader("Carga tu documento PDF",type="pdf")

#Esta funcion crea los embedings
def crear_embeddings(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000,chunk_overlap = 100)
    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_texts(chunks,embeddings)
    #Retorna la data
    return db

#Si se carga el archivo comienza a crear los embedings
if pdf_obj:
    db = crear_embeddings(pdf_obj)
    #Crea el imput para poder pedir la pregunta
    question_user = st.text_input("Realiza una pregunta...")

    #Si ya se manda la pregunta empieza a trabajar para buscar la respuesta
    if question_user:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_KHqH"
        docs = db.similarity_search(question_user,3)
        llm = HuggingFaceHub(repo_id="google/flan-t5-base")
        chain = load_qa_chain(llm,chain_type="stuff")
        respuesta = chain.run(input_documents = docs, question = question_user)
        #muestra la respuesta
        st.write(respuesta)

