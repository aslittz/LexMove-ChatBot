"""
LexMove Chatbot - Data Loader Module
====================================
Bu modül, Türk mevzuat verilerini Hugging Face'ten yükler, temizler ve
RAG sistemine hazır hale getirir.

Modül İçeriği:
- load_dataset(): Veri setini Hugging Face'ten yükler ve RAG formatına dönüştürür
- clean_text(): Metin temizleme ve normalizasyon
- chunk_text(): Metinleri parçalara böler (chunking)
- prepare_dataset(): Tüm veri hazırlama pipeline'ı
- save_processed_data(): İşlenmiş veriyi kaydeder
"""

import re
import logging
import pandas as pd
from typing import List, Optional
from tqdm import tqdm

# Metin parçalama için gerekli kütüphane (install edilmeli)
from langchain.text_splitter import RecursiveCharacterTextSplitter 

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# tqdm'i pandas'a entegre et
tqdm.pandas()


def load_dataset(dataset_path: str = "hf://datasets/muhammetakkurt/mevzuat-gov-dataset/mevzuat_parsed.json") -> pd.DataFrame:
    """
    Hugging Face'ten mevzuat veri setini yükler ve RAG formatına dönüştürür.
    """
    try:
        logger.info(f"Veri seti yükleniyor: {dataset_path}")

        # Veri setini yükle
        df = pd.read_json(dataset_path)
        logger.info(f"Toplam {len(df)} kayıt yüklendi")

        # Veri setini RAG için beklenen formata dönüştür
        logger.info("Veri setini RAG formatına dönüştürüyor: text çıkarma...")

        # 1. Başlık, ID ve Tarih Sütunlarını Eşleştir
        df['id'] = df['kanun_numarasi'].astype(str) + '_' + df['kabul_tarihi'] 
        df['title'] = df['Kanun Adı']
        df['date'] = df['kabul_tarihi']

        # 2. Maddeler sütunundan metinleri birleştirerek 'text' sütununu oluştur
        def extract_and_join_text(maddeler_list):
            if isinstance(maddeler_list, list):
                full_text = " ".join([m.get('text', '') for m in maddeler_list])
                return full_text
            return ""

        # Metin çıkarma işlemini uygula
        df['text'] = df['maddeler'].progress_apply(extract_and_join_text)
        
        # 3. Sadece RAG için gerekli nihai sütunları seç
        df = df[['id', 'title', 'text', 'date']]
        
        logger.info("Veri seti RAG formatına dönüştürüldü.")
        return df
        
    except Exception as e:
        logger.error(f"Veri yüklenirken veya dönüştürülürken hata oluştu: {e}")
        raise


def clean_text(text: str) -> str:
    """Metin temizleme ve normalizasyon işlemleri yapar."""
    if not isinstance(text, str):
        return ""
    
    # Fazla boşlukları ve yeni satırları temizle
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Bazı özel karakterleri temizle (isteğe bağlı)
    # text = text.replace('...', '')
    
    return text

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """LangChain RecursiveCharacterTextSplitter kullanarak metni parçalara ayırır."""
    if not text:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # Bölme öncelikleri
    )
    
    # Parçalama ve temizleme
    chunks = [clean_text(chunk.page_content) for chunk in text_splitter.create_documents([text]) if chunk.page_content]
    
    return chunks


def prepare_dataset(df: pd.DataFrame, chunk_size: int = 500, overlap: int = 50, show_progress: bool = True) -> pd.DataFrame:
    """
    Tüm veri hazırlama pipeline'ını çalıştırır: Temizleme, Parçalama (Chunking) ve Çerçeveleme.
    """
    logger.info("Veri hazırlama pipeline'ı başlatıldı (Chunking).")

    # Geçici bir kopyasını al
    df_temp = df.copy()

    # Eğer text sütunu yoksa veya boşsa, chunking yapılamaz
    if 'text' not in df_temp.columns or df_temp['text'].isnull().all():
        logger.error("Text sütunu bulunamadı veya boş. Veri yükleme başarısız olmuş olabilir.")
        return pd.DataFrame()

    # 1. Metin Temizleme
    logger.info("Metin temizleme uygulanıyor...")
    df_temp['text'] = df_temp['text'].progress_apply(clean_text)

    # 2. Metin Parçalama (Chunking)
    logger.info("Metin parçalama (chunking) uygulanıyor...")

    # Her satırdaki metni parçalara ayır ve yeni bir liste sütunu oluştur
    if show_progress:
        chunks_list = df_temp['text'].progress_apply(
            lambda x: chunk_text(x, chunk_size, overlap)
        ).tolist()
    else:
        chunks_list = df_temp['text'].apply(
            lambda x: chunk_text(x, chunk_size, overlap)
        ).tolist()


    # 3. Genişletme (Explode): Her parçayı ayrı bir satıra dönüştür
    # Önce chunk listelerini df'e ekle
    df_temp['chunks'] = chunks_list

    # explode ile her parçayı ayrı bir satıra yay
    df_chunks = df_temp.explode('chunks').rename(columns={'chunks': 'chunk_text'}).reset_index(drop=True)

    # Orijinal metni ve boş chunkları temizle
    df_chunks = df_chunks[df_chunks['chunk_text'].str.strip().astype(bool)].drop(columns=['text', 'maddeler'], errors='ignore')
    
    # Chunk ID ve index oluştur
    df_chunks['chunk_index'] = df_chunks.groupby('id').cumcount()
    df_chunks['chunk_id'] = df_chunks['id'].astype(str) + '_' + df_chunks['chunk_index'].astype(str)

    logger.info(f"Chunking tamamlandı. Toplam {len(df_chunks)} parça oluşturuldu.")
    return df_chunks


def save_processed_data(df: pd.DataFrame, output_path: str, save_format: str = "csv"):
    """
    İşlenmiş veri çerçevesini belirtilen formata (CSV, Parquet) kaydeder.
    """
    logger.info(f"İşlenmiş veri kaydediliyor: {output_path}")

    if save_format.lower() == "csv":
        df.to_csv(output_path, index=False)
    elif save_format.lower() == "parquet":
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Desteklenmeyen kaydetme formatı: {save_format}")

    logger.info("Veri kaydetme başarılı.")


def get_dataset_statistics(df: pd.DataFrame) -> dict:
    """Chunk boyutu istatistiklerini hesaplar."""
    if df.empty or 'chunk_text' not in df.columns:
        return {"Hata": "Veri çerçevesi boş veya chunk_text sütunu yok."}
    
    df['chunk_len'] = df['chunk_text'].apply(len)
    
    stats = {
        "Toplam Chunk Sayısı": len(df),
        "Ortalama Chunk Uzunluğu (Karakter)": int(df['chunk_len'].mean()),
        "Minimum Chunk Uzunluğu (Karakter)": df['chunk_len'].min(),
        "Maksimum Chunk Uzunluğu (Karakter)": df['chunk_len'].max(),
        "Benzersiz Doküman Sayısı (id)": df['id'].nunique()
    }
    return stats
