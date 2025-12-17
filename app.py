import os
import sys
import subprocess
import warnings
import streamlit as st
from pathlib import Path
from typing import List

warnings.filterwarnings('ignore')

print("Modüller yükleniyor...")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    from langchain_community.chat_models import ChatGoogleGenerativeAI

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    from langchain.schema import Document
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser

print("Tüm modüller yüklendi")


st.set_page_config(
    page_title="LexMove RAG Chatbot", 
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("LexMove Hukuk Chatbotu")
st.markdown("Türk Hukuku (Mini Q&A) veri seti ile desteklenen yapay zeka danışmanı.")



BASE_DIR = Path(__file__).parent.absolute()
CHROMA_PATH = BASE_DIR / "chroma_db_lexmove_mini"
COLLECTION_NAME = "mevzuat_chunks_mini" 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@st.cache_resource(show_spinner=False)
def setup_database_if_needed():
    """ChromaDB yoksa otomatik oluştur"""
    

    if CHROMA_PATH.exists():
        try:
            files = list(CHROMA_PATH.glob("*"))
            if len(files) > 0:
                return True  
        except:
            pass
    

    st.info(" İlk kurulum yapılıyor... (Bu işlem 2-3 dakika sürebilir)")
    st.info(" Hugging Face'ten veri seti indiriliyor ve işleniyor...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("⏳ setup_database.py çalıştırılıyor...")
        progress_bar.progress(25)
        

        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "setup_database.py")],
            capture_output=True,
            text=True,
            check=True
        )
        
        progress_bar.progress(75)
        status_text.text(" Database oluşturuldu, yükleniyor...")
        
        # Başarı kontrolü
        if CHROMA_PATH.exists() and len(list(CHROMA_PATH.glob("*"))) > 0:
            progress_bar.progress(100)
            status_text.empty()
            st.success("Database başarıyla oluşturuldu!")
            st.balloons()
            return True
        else:
            st.error(" Database oluşturulamadı!")
            return False
            
    except subprocess.CalledProcessError as e:
        progress_bar.empty()
        status_text.empty()
        st.error(" Database kurulum hatası!")
        st.error(f"Detay: {e.stderr}")
        st.info(" Lütfen sayfayı yenileyin ve tekrar deneyin.")
        return False
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f" Beklenmeyen hata: {str(e)}")
        return False



@st.cache_resource
def get_vectorstore():
    """ChromaDB vektör veritabanını yükler"""
    
  
    if not CHROMA_PATH.exists():
        st.error(f" HATA: ChromaDB klasörü bulunamadı!")
        st.error(f" Aranan yer: {CHROMA_PATH}")
        return None
    

    try:
        files_in_chroma = list(CHROMA_PATH.glob("*"))
        if len(files_in_chroma) == 0:
            st.error(" ChromaDB klasörü boş!")
            return None
    except Exception as e:
        st.error(f" Klasör okunamadı: {e}")
        return None
    
    try:

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        

        vectorstore = Chroma(
            persist_directory=str(CHROMA_PATH),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        
        # Vektör sayısını al
        try:
            chunk_count = vectorstore._collection.count()
        except:
            chunk_count = 0
        
        if chunk_count == 0:
            st.error(" ChromaDB boş! Hiç vektör bulunamadı.")
            return None
        
        return vectorstore
        
    except Exception as e:
        st.error(f" ChromaDB bağlantı hatası:")
        st.exception(e)
        return None

@st.cache_resource
def get_llm():
    """Google Gemini LLM'i başlatır"""
    
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except:
        # Local .env'den oku
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        st.error(" HATA: GOOGLE_API_KEY bulunamadı!")
        st.info("Streamlit Cloud için:")
        st.code("App Settings → Secrets → GOOGLE_API_KEY ekleyin", language='text')
        st.info("Yerel çalışma için:")
        st.code('GOOGLE_API_KEY=your_api_key_here', language='bash')
        st.info("API anahtarı almak için: https://ai.google.dev/")
        return None
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.3,
            max_output_tokens=2048,
            google_api_key=api_key
        )
        return llm
    except Exception as e:
        st.error(f"LLM başlatma hatası:")
        st.exception(e)
        st.info(" API anahtarınızın geçerli olduğundan emin olun")
        return None



with st.sidebar:
    st.markdown("###  Sistem Durumu")
    
    if CHROMA_PATH.exists() and len(list(CHROMA_PATH.glob("*"))) > 0:
        st.success(" Database aktif")
    else:
        st.warning(" Database kurulumu gerekli")

with st.spinner(" Sistem hazırlanıyor..."):
    db_ready = setup_database_if_needed()
    
    if not db_ready:
        st.error(" Sistem başlatılamadı. Lütfen sayfayı yenileyin.")
        st.stop()

# 
with st.spinner("Bileşenler yükleniyor..."):
    vectorstore = get_vectorstore()
    llm = get_llm()


with st.sidebar:
    if vectorstore:
        try:
            chunk_count = vectorstore._collection.count()
            st.metric(" Toplam Vektör", f"{chunk_count:,}")
        except:
            pass
    
    if llm:
        st.success("LLM hazır (Gemini 2.0)")
    else:
        st.error(" LLM başlatılamadı")



def format_docs(docs: List[Document]) -> str:
    """Çekilen document listesini tek bir string bağlamına dönüştürür."""
    if not docs:
        return "İlgili bilgi bulunamadı."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def retrieve_relevant_chunks(query: str, k: int = 8) -> List[Document]:
    """Sorguya göre ChromaDB'den en alakalı parçaları çeker."""
    if vectorstore is None:
        return []
    try:
        return vectorstore.similarity_search(query=query, k=k)
    except Exception as e:
        st.error(f" Vektör arama hatası: {e}")
        return []

RAG_PROMPT_TEMPLATE = """
Sen, Türk hukuku konusunda uzmanlaşmış, yapay zeka destekli bir danışmansın.
Görevinde sadece, sana aşağıda sağlanan 'BAĞLAM' içerisindeki bilgileri kullanmalısın.
Eğer BAĞLAM'daki bilgiler soruyu yanıtlamak için yeterli değilse, kesinlikle uydurma yapma veya genel bilgi verme.
Böyle bir durumda cevabın "Maalesef elimdeki mevzuat metinlerinde sorunuzun cevabına uygun bir yanıt bulamadım..." olmalıdır.

Aşağıdaki kurallara kesinlikle uy:
1. Cevaplarını Türkçe ve kibar bir dille ver.
2. Cevabını BAĞLAM'daki bilgilere dayandırarak oluştur.
3. Yanıtın, çekilen BAĞLAM'daki en detaylı ve açıklayıcı bilgi parçasını içermeli ve bu bilgiyi olabildiğince eksiksiz sunmalıdır.
4. Çekilen BAĞLAM'daki hukuki açıklamanın tamamını kullanmaya çalış.
5. Kısa ve net cevaplar ver, gereksiz tekrar yapma.

BAĞLAM:
{context}

KULLANICININ SORUSU:
{question}

SENİN YANITIN:
"""

RAG_PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

if llm and vectorstore:
    rag_chain = (
        {
            "context": lambda x: format_docs(retrieve_relevant_chunks(x["question"])),
            "question": lambda x: x["question"]
        }
        | RAG_PROMPT 
        | llm 
        | StrOutputParser()
    )
else:
    rag_chain = None




if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant", 
            "content": "Merhaba! Türk hukuku ile ilgili sorularınızı cevaplayabilirim. Nasıl yardımcı olabilirim? 📚"
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input(placeholder="Hukuki sorunuzu buraya yazın...")

if user_input:
    if not rag_chain:
        st.error(" Sistem henüz hazır değil. Lütfen yukarıdaki hataları kontrol edin.")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner(" Mevzuat taranıyor..."):
            try:
                response = rag_chain.invoke({"question": user_input})
                
                st.write(response)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                
            except Exception as e:
                error_msg = f" Bir hata oluştu: {str(e)}"
                st.error(error_msg)
                st.info("Birkaç saniye bekleyip tekrar deneyin veya sorunuzu yeniden ifade edin.")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })


with st.sidebar:
    st.markdown("---")
    st.markdown("###  Kullanım İpuçları")
    st.markdown("""
    - Türk hukuku hakkında soru sorun
    - Net ve spesifik sorular sorun
    - Sistem sadece veri setindeki bilgileri kullanır
    """)
    
    st.markdown("---")
    st.markdown("###  Proje Bilgileri")
    st.markdown("""
    **LexMove RAG Chatbot**
    
    -  Model: Gemini 2.0 Flash
    -  Vector DB: ChromaDB
    -  Embedding: all-MiniLM-L6-v2
    -  Dataset: turkish-law-chatbot
    """)
    
    st.markdown("---")
    st.markdown("###  Akbank GenAI Bootcamp")
    st.markdown("*Yeni Nesil Proje Kampı*")
    
    if st.button(" Sohbeti Temizle"):
        st.session_state["messages"] = [
            {
                "role": "assistant", 
                "content": "Merhaba! Türk hukuku ile ilgili sorularınızı cevaplayabilirim. Nasıl yardımcı olabilirim? 📚"
            }
        ]
        st.rerun()
    
    st.markdown("---")
    st.markdown("[GitHub](https://github.com/yourusername/lexmove-rag)")
    st.caption("Made with  for Akbank Bootcamp")
