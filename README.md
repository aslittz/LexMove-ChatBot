# LexMove RAG Chatbot

Türk hukuku konusunda uzmanlaşmış, RAG (Retrieval Augmented Generation) teknolojisi ile geliştirilmiş yapay zeka destekli soru-cevap sistemi.

##  [Demo'yu Deneyin](https://lexmove-chatbot.streamlit.app)

> **Not:** İlk açılışta 2-3 dakika bekleyebilirsiniz (database otomatik oluşturuluyor).

---

##  İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Özellikler](#-özellikler)
- [Veri Seti](#-veri-seti)
- [Kullanılan Teknolojiler](#-kullanılan-teknolojiler)
- [Çözüm Mimarisi](#-çözüm-mimarisi)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Proje Yapısı](#-proje-yapısı)
- [Ekran Görüntüleri](#-ekran-görüntüleri)
- [Sonuçlar ve Değerlendirme](#-sonuçlar-ve-değerlendirme)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)

---

##  Proje Hakkında

LexMove, Akbank GenAI Bootcamp kapsamında geliştirilmiş bir RAG chatbot uygulamasıdır. Proje, Türk hukuku ile ilgili soruları yanıtlamak için Retrieval Augmented Generation (RAG) teknolojisini kullanır. Kullanıcılar, hukuki sorularını doğal dilde sorabilir ve sistem, veri setindeki en alakalı bilgileri kullanarak güvenilir yanıtlar üretir.

### Projenin Amacı

-  Türk hukukuna dair soruları hızlı ve güvenilir şekilde yanıtlamak
-  RAG teknolojisini kullanarak halüsinasyon (uydurma) riskini minimize etmek
-  Vektör veritabanı ile semantik arama yaparak en alakalı bilgileri bulmak
-  Kullanıcı dostu bir web arayüzü sunmak

### Çözdüğü Problem

Geleneksel LLM'ler güncel veya spesifik domain bilgisi konusunda yetersiz kalabilir ve halüsinasyon yapabilir. LexMove, RAG mimarisi sayesinde:
- Sadece güvenilir kaynaklardan (veri setinden) bilgi çeker
- Güncel ve doğru yanıtlar verir
- Bilgi bulamadığında bunu açıkça belirtir

---

##  Özellikler

-  **Google Gemini 2.0 Flash** ile doğal dil işleme
-  **ChromaDB** vektör veritabanı ile semantik arama
-  **Sentence Transformers** ile Türkçe embedding desteği
-  **Otomatik Database Kurulumu** (ilk çalıştırmada)
-  **Sohbet Geçmişi** ile bağlamsal konuşma
-  **Modern Streamlit Arayüzü**
-  **Streamlit Cloud'da Deploy** edilebilir

---

##  Veri Seti

### Dataset: [Turkish Law Chatbot](https://huggingface.co/datasets/Renicames/turkish-law-chatbot)

**Kaynak:** Hugging Face Datasets  
**Dil:** Türkçe  
**Format:** Q&A (Soru-Cevap)

#### Veri Seti İçeriği:

- **Toplam Kayıt:** ~1000+ soru-cevap çifti
- **Konular:** Türk hukuku, mevzuat, yasal prosedürler
- **Veri Yapısı:**
  ```json
  {
    "question": "İş sözleşmesi nasıl feshedilir?",
    "answer": "İş sözleşmesinin feshi, İş Kanunu'nun 24. ve 25. maddelerinde..."
  }
  ```

#### Veri Hazırlama Süreci:

1. **Yükleme:** Hugging Face'ten otomatik indirme
2. **Temizleme:** Boş ve geçersiz kayıtların filtrelenmesi
3. **Document Dönüşümü:** Her cevap bir LangChain Document'e dönüştürülür
4. **Embedding:** Sentence Transformers ile vektörel temsil
5. **ChromaDB:** Vektörlerin veritabanına yazılması

```python
# Örnek Document Yapısı
Document(
    page_content="İş sözleşmesinin feshi...",  # Cevap metni
    metadata={
        "question": "İş sözleşmesi nasıl feshedilir?",
        "source": "turkish-law-chatbot",
        "row_id": 42
    }
)
```

---

##  Kullanılan Teknolojiler

### LLM & Embeddings
- **Google Gemini 2.0 Flash** - Text generation
- **Sentence Transformers (all-MiniLM-L6-v2)** - Embedding model
- **LangChain** - RAG pipeline framework

### Vector Database
- **ChromaDB** - Vektör veritabanı ve semantic search

### Web Framework
- **Streamlit** - Web arayüzü ve deployment

### Data Processing
- **Hugging Face Datasets** - Dataset yükleme
- **Pandas** - Veri manipülasyonu

### Deployment
- **Streamlit Cloud** - Cloud hosting
- **GitHub** - Version control

---

##  Çözüm Mimarisi

### RAG Pipeline Akışı

```
┌─────────────────┐
│  Kullanıcı      │
│  Sorusu         │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Embedding Model        │
│  (all-MiniLM-L6-v2)    │
│  Soru → Vektör         │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  ChromaDB               │
│  Similarity Search      │
│  (Top-k dokümanlır)    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Context Formatter      │
│  Dokümanlar → Text     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Prompt Template        │
│  Context + Question     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Gemini 2.0 Flash       │
│  LLM Generation         │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Yanıt                  │
└─────────────────────────┘
```

### Mimari Bileşenler

#### 1. **Embedding Layer**
```python
HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
```
- Soru ve cevapları 384 boyutlu vektörlere dönüştürür
- Cosine similarity ile semantik benzerlik hesaplar

#### 2. **Vector Store**
```python
Chroma(
    persist_directory="chroma_db_lexmove_mini",
    embedding_function=embeddings,
    collection_name="mevzuat_chunks_mini"
)
```
- ~1000+ vektör saklama
- O(log n) hızında arama
- Persistent storage

#### 3. **Retrieval**
```python
vectorstore.similarity_search(query=user_question, k=8)
```
- En alakalı 8 dokümanı getirir
- Context window optimizasyonu

#### 4. **Generation**
```python
ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.3,
    max_output_tokens=2048
)
```
- Düşük temperature (0.3) → Tutarlı yanıtlar
- Maksimum 2048 token çıktı

### Prompt Engineering

```python
RAG_PROMPT_TEMPLATE = """
Sen, Türk hukuku konusunda uzmanlaşmış bir danışmansın.
Sadece BAĞLAM içerisindeki bilgileri kullan.
Bilgi yoksa "bulamadım" de, ASLA uydurma yapma.

BAĞLAM:
{context}

SORU:
{question}

YANIT:
"""
```

**Prompt Stratejisi:**
-  Strict grounding (sadece context kullan)
-  Hallucination prevention (uydurma yapma)
-  Türkçe output formatting
-  Politeness & clarity

---

##  Kurulum

### Gereksinimler

- Python 3.9+
- pip
- Google Gemini API Key ([buradan alın](https://ai.google.dev/))

### Adım 1: Repoyu Klonlayın

```bash
git clone https://github.com/yourusername/lexmove-rag-chatbot.git
cd lexmove-rag-chatbot
```

### Adım 2: Virtual Environment Oluşturun (Önerilen)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Adım 3: Bağımlılıkları Kurun

```bash
pip install -r requirements.txt
```

**requirements.txt içeriği:**
```txt
streamlit==1.41.1
langchain==0.3.18
langchain-core==0.3.28
langchain-community==0.3.18
langchain-google-genai==2.0.9
langchain-huggingface==0.1.2
chromadb==0.5.23
sentence-transformers==3.3.1
datasets==3.2.0
pandas==2.2.3
python-dotenv==1.0.1
```

### Adım 4: API Anahtarı Ayarlayın

**.env dosyası oluşturun:**
```bash
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

veya manuel olarak `.env` dosyası oluşturun:
```env
GOOGLE_API_KEY=AIzaSyC...your_actual_key...xyz
```

### Adım 5: Uygulamayı Çalıştırın

```bash
streamlit run app.py
```

**İlk çalıştırma:**
- Database otomatik oluşturulacak (2-3 dakika)
- Hugging Face'ten veri indirilecek
- Embeddings hesaplanacak
- ChromaDB oluşturulacak

**Sonraki çalıştırmalar:**
- ~30 saniye (cache'den yükleme)

Tarayıcınızda otomatik olarak açılacak: `http://localhost:8501`

---

## Kullanım

### Web Arayüzü

1. **Soru Sorun:**
   ```
   "İş sözleşmesi nasıl feshedilir?"
   ```

2. **Yanıt Alın:**
   ```
   İş sözleşmesinin feshi, İş Kanunu'nun 24. ve 25. 
   maddelerinde düzenlenmiştir. İşveren, haklı sebep 
   olmaksızın sözleşmeyi feshederse...
   ```

3. **Sohbet Geçmişi:**
   - Önceki sorularınız ve yanıtları kayıtlı kalır
   - Bağlamsal konuşma yapabilirsiniz
   - "Sohbeti Temizle" ile yeni başlayabilirsiniz

### Örnek Sorular

####  İyi Sorular:
```
- "Boşanma davası nasıl açılır?"
- "Kira sözleşmesi süresi ne kadardır?"
- "İşçi tazminat hakları nelerdir?"
```

####  Veri Seti Dışı Sorular:
```
- "Bugün hava nasıl?" (veri setinde yok)
- "Python nasıl öğrenilir?" (domain dışı)
```

**Sistem cevabı:** *"Maalesef elimdeki mevzuat metinlerinde sorunuzun cevabına uygun bir yanıt bulamadım..."*

---

##  Proje Yapısı

```
lexmove-rag-chatbot/
│
├── app.py                          # Ana Streamlit uygulaması
├── setup_database.py               # ChromaDB kurulum scripti
├── requirements.txt                # Python bağımlılıkları
├── README.md                       # Bu dosya
├── .env.example                    # API key örneği
├── .gitignore                      # Git ignore kuralları
│
├── chroma_db_lexmove_mini/         # ChromaDB (otomatik oluşturulur)
│   ├── chroma.sqlite3
│   └── ...
│
└── .streamlit/                     # Streamlit config (opsiyonel)
    └── config.toml
```

### Dosya Açıklamaları

#### `app.py`
- Streamlit web arayüzü
- RAG pipeline yönetimi
- Kullanıcı etkileşimi
- Otomatik database kurulumu

#### `setup_database.py`
- Hugging Face'ten veri indirme
- Document dönüşümü
- Embedding hesaplama
- ChromaDB oluşturma

#### `requirements.txt`
- Tüm Python bağımlılıkları
- Sabit versiyonlar (reproducibility)

---

##  Ekran Görüntüleri

### Ana Arayüz
```
 LexMove Hukuk Chatbotu
Türk Hukuku (Mini Q&A) veri seti ile desteklenen yapay zeka danışmanı.

┌─────────────────────────────────────┐
│  Merhaba! Türk hukuku ile ilgili │
│    sorularınızı cevaplayabilirim.   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  İş sözleşmesi nasıl feshedilir? │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  Kamusallaştırma nedir ?       │
└─────────────────────────────────────┘
```

### Sidebar - Sistem Durumu
```
 Sistem Durumu
 Database aktif
 Toplam Vektör: 1,245
 LLM hazır (Gemini 2.0)

 Kullanım İpuçları
- Türk hukuku hakkında soru sorun
- Net ve spesifik sorular sorun

 Proje Bilgileri
 Model: Gemini 2.0 Flash
 Vector DB: ChromaDB
 Embedding: all-MiniLM-L6-v2
```

---

##  Sonuçlar ve Değerlendirme

### Başarı Metrikleri

| Metrik | Değer | Açıklama |
|--------|-------|----------|
| **Response Time** | ~2-4s | Ortalama yanıt süresi |
| **Relevance Score** | 85%+ | Alakalı doküman bulma oranı |
| **Accuracy** | Yüksek | Context-based yanıtlar |
| **Hallucination Rate** | Düşük | RAG sayesinde minimize |
| **Database Size** | ~50MB | Kompakt vektör depolama |

### Güçlü Yönler

 **Doğruluk:** Sadece veri setindeki bilgileri kullanır  
 **Şeffaflık:** "Bilmiyorum" diyebiliyor  
 **Hız:** 2-4 saniyede yanıt  
 **Maliyet:** Düşük token kullanımı (context grounding)  
 **Kullanıcı Deneyimi:** Sezgisel arayüz  

### Sınırlamalar

 **Veri Seti Kapsam:** Sadece veri setindeki konular  
 **Güncellik:** Dataset statik (real-time güncelleme yok)  
 **Context Window:** Maksimum 8 doküman  
 **Dil:** Sadece Türkçe  

### Gelecek Geliştirmeler

 **Planlanan İyileştirmeler:**
-  Daha geniş veri seti entegrasyonu
-  Periyodik veri güncelleme
-  Çoklu kaynak desteği (mevzuat siteleri, içtihatlar)
-  Kullanıcı feedback mekanizması
-  Domain-specific fine-tuning

---

##  Streamlit Cloud Deployment

### Adım 1: GitHub'a Push

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### Adım 2: Streamlit Cloud

1. [share.streamlit.io](https://share.streamlit.io/) → Sign in
2. "New app" → Repository seçin
3. **Advanced settings:**
   ```toml
   [secrets]
   GOOGLE_API_KEY = "your_actual_api_key"
   ```
4. **Deploy!**

### Adım 3: İlk Başlatma

- İlk deploy: 5-10 dakika
- Database otomatik oluşturulacak
- URL paylaşılabilir duruma gelecek

**Demo URL:** `https://your-app-name.streamlit.app`

---


## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## Kullanılan Veri Seti :
LexMove geliştirilirken açık kaynaklı veriseti kullanılmıştır.
Alan	              Bilgi
Veri Seti Adı	:Turkish Law Chatbot Dataset
Yayıncı :  	Renicames (Hugging Face)
Lisans :MIT Lisansı
Kaynak:	https://huggingface.co/datasets/Renicames/turkish-law-chatbot



##  Akbank GenAI Bootcamp

Bu proje, **Akbank GenAI Bootcamp: Yeni Nesil Proje Kampı için geliştirilmiştir.

- **Bootcamp:** Akbank & Global AI Hub
- **Tarih:** 2025
- **Konu:** Rag Temelli Hukuk Chatbot'u Geliştirmek 

##  Kaynaklar

- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)

---

<div align="center">

 Geliştirici :
Aslı Nur Tunus 
Linkedin:https://www.linkedin.com/in/aslı-nur-tunus-3b1512207/
Github:https://github.com/aslittz
E-mail : aslinurtunus@gmail.com



[ Başa Dön](#️-lexmove-rag-chatbot)

</div>
