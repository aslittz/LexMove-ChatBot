# ================================================================
# STREAMLIT CHATBOT UYGULAMASI (app.py)
# ================================================================

import os
import streamlit as st
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv, find_dotenv

# LangChain bileşenleri
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ----------------------------------------------------
# 1. AYARLAR VE VERİTABANI BAĞLANTISI (Notebook 3'ten)
# ----------------------------------------------------

# API Key yükle (Streamlit'te bu adımı yapmayız, Streamlit Secrets kullanırız, ancak yerelde tutarlılık için kalsın)
load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=False))

# ChromaDB Ayarları
CHROMA_PATH = os.path.expanduser("~/chroma_db_lexmove_mini")
COLLECTION_NAME = "mevzuat_chunks_mini"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Embedding Modelini ve Vektör Veritabanını Tek Bir Kez Yükle
# Streamlit'in önbellek (cache) mekanizması kullanılır
@st.cache_resource
def get_vectorstore():
    # Embedding Model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    try:
        # ChromaDB Bağlantısı
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        # st.success(f"ChromaDB yüklendi. Toplam parça: {vectorstore._collection.count()}")
        return vectorstore
    except Exception as e:
        st.error(f"❌ ChromaDB bağlantı hatası: {e}. Lütfen yolu kontrol edin.")
        return None

vectorstore = get_vectorstore()
current_api_key = os.getenv('GOOGLE_API_KEY') # Tek anahtar kullanımı varsayılır

# LLM Tanımı (Tek anahtar ile)
LLM = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    max_output_tokens=1024,
    google_api_key=current_api_key
)

# ----------------------------------------------------
# 2. RAG FONKSİYONLARI (Notebook 3'ten)
# ----------------------------------------------------

def format_docs(docs: List[Document]) -> str:
    """Çekilen document listesini tek bir string bağlamına dönüştürür."""
    # Q&A içeriğinde yalnızca page_content'ı birleştiriyoruz.
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def retrieve_relevant_chunks(query: str, db: Chroma, k: int = 10) -> List[Document]:
    """Sorguya göre ChromaDB'den en alakalı parçaları çeker."""
    if db is None:
        return []
    # k=10, daha tutarlı cevaplar için Notebook 3'teki revizyonumuz
    return db.similarity_search(query=query, k=k)

# Prompt Şablonu (Notebook 3'teki revize edilmiş, detaycı şablon)
RAG_PROMPT_TEMPLATE = """
Sen, Türk hukuku konusunda uzmanlaşmış, yapay zeka destekli bir danışmansın.
Görevinde sadece, sana aşağıda sağlanan 'BAĞLAM' içerisindeki bilgileri kullanmalısın.
Eğer BAĞLAM'daki bilgiler soruyu yanıtlamak için yeterli değilse, kesinlikle uydurma yapma veya genel bilgi verme.
Böyle bir durumda cevabın "Üzgünüm, bu soruya yanıt verebilecek yeterli ve doğrulanmış hukuki bilgiye BAĞLAM'da ulaşamadım." olmalıdır.

Aşağıdaki kurallara kesinlikle uy:
1. Cevaplarını Türkçe ve kibar bir dille ver.
2. Cevabını BAĞLAM'daki bilgilere dayandırarak oluştur.
3. Yanıtın, çekilen BAĞLAM'daki en detaylı ve açıklayıcı bilgi parçasını içermeli ve bu bilgiyi olabildiğince eksiksiz sunmalıdır.
4. Çekilen BAĞLAM'daki hukuki açıklamanın tamamını kullanmaya çalış.

BAĞLAM:
{context}

KULLANICININ SORUSU:
{question}

SENİN YANITIN:
"""
RAG_PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# RAG Zinciri
rag_chain = (
    {"context": lambda x: format_docs(retrieve_relevant_chunks(x["question"], vectorstore)),
     "question": lambda x: x["question"]}
    | RAG_PROMPT 
    | LLM 
    | StrOutputParser()
)

# ----------------------------------------------------
# 3. STREAMLIT ARAYÜZÜ
# ----------------------------------------------------

st.set_page_config(page_title="LexMove RAG Chatbot", layout="centered")

# Başlıklar
st.title("⚖️ LexMove Hukuk Chatbotu")
st.markdown("Türk Hukuku (Mini Q&A) veri seti ile desteklenen yapay zeka danışmanı.")


# Streamlit Session State kullanarak sohbet geçmişini koruma
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Merhaba! Türk hukuku ile ilgili sorularınızı cevaplayabilirim. Nasıl yardımcı olabilirim?"}
    ]

# Sohbet geçmişini gösterme
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Kullanıcı girişi
if prompt := st.chat_input(placeholder="Hukuki sorunuzu buraya yazın..."):
    # Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Yanıtı al
    with st.spinner("Cevap hazırlanıyor..."):
        try:
            # RAG Zincirini çağır
            response = rag_chain.invoke({"question": prompt})
            
            # Asistan yanıtını ekle ve göster
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
            
        except Exception as e:
            # Hata durumunda (Örn: API limit) kullanıcıya bilgi ver
            error_message = f"API Limit Aşımı veya Hata: Lütfen 5 saniye bekleyip tekrar deneyin. Detay: {e.__class__.__name__}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.chat_message("assistant").error(error_message)