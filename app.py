# Importación de Las librerias
from bs4 import BeautifulSoup #Libreria para trabajar con html
from langchain.text_splitter import RecursiveCharacterTextSplitter #Libreria para Splitear la informacion
from langchain.embeddings import HuggingFaceEmbeddings #Librearia para crear los embeddings
from langchain.vectorstores import FAISS #libreria para crear la data
from langchain.chains.question_answering import load_qa_chain #Libreria para cargar el modelo LLM
from langchain import HuggingFaceHub  #Libreraia para usar los modelos de hugginface
import os #libreria para usar el token de hugginFace
import streamlit as st #Necesaria para poder realizar la interfaz local
import io


#Coloca el titulo de la pagina
st.title('Preguntar por po tu pagina web')
#Necesario para subir el archivo
html_obj= st.file_uploader("Carga tu archivo HTML", type="html")


#Esta funcion crea los embeddings
def crear_embeddings(html):
    # Apertura del archivo HTML en modo lectura y lectura de su contenido
    with io.BytesIO(html.read()) as html_file:
        html_content = html_file.read().decode('utf-8')

    # Creación de un objeto BeautifulSoup para analizar el contenido HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extracción del texto del contenido HTML
    text = soup.get_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 100)
    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_texts(chunks, embeddings)
    #Retorna la data
    return db

#Si se carga el archivo comienza a crear los embedings
if html_obj:
    db = crear_embeddings(html_obj)
    #Crea el imput para poder pedir la pregunta
    question_user = st.text_input("Realiza una pregunta...")

    #Si ya se manda la pregunta empieza a trabajar para buscar la respuesta
    if question_user:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_UNdn"
        docs = db.similarity_search(question_user)
        llm = HuggingFaceHub(repo_id="google/flan-t5-base")
        chain = load_qa_chain(llm, chain_type="stuff")
        respuesta = chain.run(input_documents = docs, question = question_user)
        #muestra la respuesta
        st.write(respuesta)