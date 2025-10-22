# ================================================================
# setup_database.py — HUGGING FACE DATASET VERSİYONU (TAM TEST EDİLMİŞ)
# ================================================================
import os
import shutil
from pathlib import Path
from typing import List

print("📦 Modüller yükleniyor...")

try:
    import pandas as pd
    from datasets import load_dataset
except ImportError as e:
    print(f"❌ HATA: {e}")
    print("💡 Şu komutu çalıştırın: pip install pandas datasets")
    exit(1)

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    try:
        from langchain.vectorstores import Chroma
    except ImportError as e:
        print(f"❌ HATA: {e}")
        print("💡 Şu komutu çalıştırın: pip install langchain langchain-community")
        exit(1)

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
    except ImportError as e:
        print(f"❌ HATA: {e}")
        print("💡 Şu komutu çalıştırın: pip install langchain-huggingface sentence-transformers")
        exit(1)

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError as e:
        print(f"❌ HATA: {e}")
        print("💡 Şu komutu çalıştırın: pip install langchain-core")
        exit(1)

print("✅ Tüm modüller yüklendi\n")

# 🎯 PROJENİN KÖK DİZİNİ
BASE_DIR = Path(__file__).parent.absolute()

# ✅ ChromaDB'yi proje içinde tut
CHROMA_PATH = BASE_DIR / "chroma_db_lexmove_mini"

# 🔥 Hugging Face Dataset Ayarları
DATASET_NAME = "Renicames/turkish-law-chatbot"
SPLIT_NAME = "train"

COLLECTION_NAME = "mevzuat_chunks_mini"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

print("="*70)
print("🚀 LexMove - ChromaDB Kurulum Başlatıldı (Hugging Face Dataset)")
print("="*70)
print(f"📦 Dataset: {DATASET_NAME}")
print(f"📂 Çıkış dizini (Chroma): {CHROMA_PATH}")
print(f"📂 Proje kök dizini: {BASE_DIR}")

# ================================================================
# 1. HUGGING FACE'TEN VERİ YÜKLEME
# ================================================================

def load_and_create_documents() -> List[Document]:
    """
    Hugging Face Q&A veri setini yükler ve her Cevabı, Soru metadatasıyla 
    LangChain Document listesine dönüştürür.
    """
    all_documents = []
    
    print(f"\n⏳ Hugging Face'ten veri seti yükleniyor: {DATASET_NAME} ({SPLIT_NAME})")
    
    try:
        # Hugging Face'ten doğrudan yükleme
        dataset = load_dataset(DATASET_NAME, split=SPLIT_NAME)
        df = dataset.to_pandas()
        
        print(f"✅ Veri seti yüklendi. Toplam satır: {len(df)}")
        print(f"📊 Sütunlar: {list(df.columns)}")
        
    except Exception as e:
        print(f"❌ Veri seti yüklenirken HATA oluştu: {e}")
        print("💡 İnternet bağlantınızı kontrol edin veya dataset adını doğrulayın.")
        return []
    
    # Her satırı LangChain Document'e dönüştür
    for idx, row in df.iterrows():
        # Sütun isimlerini kontrol et (küçük/büyük harf duyarlı)
        question = ""
        answer = ""
        
        # Farklı olası sütun isimlerini dene
        for q_col in ['question', 'Question', 'Soru', 'soru']:
            if q_col in df.columns:
                question = str(row.get(q_col, '')).strip()
                break
        
        for a_col in ['answer', 'Answer', 'Cevap', 'cevap']:
            if a_col in df.columns:
                answer = str(row.get(a_col, '')).strip()
                break
        
        # Sadece boş olmayan cevapları ekle
        if answer and len(answer) > 10:
            doc = Document(
                page_content=answer,
                metadata={
                    "question": question,
                    "source": DATASET_NAME,
                    "row_id": int(idx)
                }
            )
            all_documents.append(doc)
    
    print(f"✅ {len(all_documents)} adet Document oluşturuldu")
    return all_documents

# ================================================================
# 2. ESKİ VERİTABANINI SİLME
# ================================================================

if CHROMA_PATH.exists():
    print(f"\n⚠️ Eski veritabanı siliniyor: {CHROMA_PATH}")
    try:
        shutil.rmtree(CHROMA_PATH)
        print("✅ Eski veritabanı silindi")
    except OSError as e:
        print(f"❌ Klasör silinirken hata: {e}")
        print("💡 Klasörü manuel olarak silmeyi deneyin veya yönetici olarak çalıştırın")
        exit(1)

# ================================================================
# 3. BELGELERİ YÜKLEME
# ================================================================

print("\n🔥 Belgeler Hugging Face'ten yükleniyor...")
documents = load_and_create_documents()

if len(documents) == 0:
    print("\n❌ HATA: Yüklenecek belge bulunamadı!")
    print("💡 Dataset boş olabilir veya sütun isimleri farklı olabilir.")
    exit(1)

print(f"\n📄 İlk belge örneği:")
print(f"  Soru: {documents[0].metadata.get('question', 'N/A')[:100]}...")
print(f"  Cevap: {documents[0].page_content[:150]}...")

# ================================================================
# 4. EMBEDDING MODELİ
# ================================================================

print("\n🔧 Embedding modeli hazırlanıyor...")
try:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("✅ Embedding modeli hazır")
except Exception as e:
    print(f"❌ Embedding modeli yüklenemedi: {e}")
    print("💡 sentence-transformers paketi kurulu mu kontrol edin")
    print("   Komut: pip install sentence-transformers")
    exit(1)

# ================================================================
# 5. CHROMADB OLUŞTURMA
# ================================================================

print("\n🧠 Chroma vektör veritabanı oluşturuluyor...")
print("⏳ Bu işlem birkaç dakika sürebilir...")

try:
    # ChromaDB dizinini oluştur
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    
    # Batch processing ile vektör oluştur (bellek tasarrufu için)
    batch_size = 100
    total_docs = len(documents)
    
    vectorstore = None
    
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i+batch_size]
        print(f"⏳ İşleniyor: {i+1}-{min(i+batch_size, total_docs)}/{total_docs}")
        
        if vectorstore is None:
            # İlk batch ile vectorstore oluştur
            vectorstore = Chroma.from_documents(
                batch, 
                embeddings,
                persist_directory=str(CHROMA_PATH), 
                collection_name=COLLECTION_NAME
            )
        else:
            # Sonraki batch'leri ekle
            vectorstore.add_documents(batch)
    
    # Vektör sayısını kontrol et
    total_chunks = vectorstore._collection.count()
    
    print("\n" + "="*70)
    print("✅ BAŞARILI! ChromaDB oluşturuldu")
    print("="*70)
    print(f"📊 Toplam vektör sayısı: {total_chunks}")
    print(f"📁 Kayıt yeri: {CHROMA_PATH}")
    print(f"📦 Koleksiyon adı: {COLLECTION_NAME}")
    print(f"🌐 Dataset: {DATASET_NAME}")
    print(f"🔍 Embedding Model: {EMBEDDING_MODEL}")
    print("\n🚀 Şimdi şu komutu çalıştırın:")
    print("   streamlit run app.py")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ HATA: ChromaDB oluşturulamadı!")
    print(f"Detay: {e}")
    print("\n💡 Olası çözümler:")
    print("  1. Klasör izinlerini kontrol edin")
    print("  2. Yeterli disk alanı olduğundan emin olun")
    print("  3. chromadb paketini kurun: pip install chromadb")
    import traceback
    traceback.print_exc()
    exit(1)