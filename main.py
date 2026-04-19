import streamlit as st
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from model_utils import load_model, predict_image
from batch_utils import process_directory, save_to_csv

# Page config
st.set_page_config(page_title="ViT Image Classifier", layout="wide")

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stAlert {
        border-radius: 10px;
    }
    .stButton>button {
        border-radius: 20px;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_cached_model():
    return load_model()

def main():
    st.title("Vision Transformer (ViT) Image Classifier")
    st.markdown("Hugging Face `google/vit-base-patch16-224` modelini kullanan profesyonel sınıflandırma aracı.")

    # Load Model
    with st.spinner("Model yükleniyor..."):
        processor, model, device = get_cached_model()

    if not model:
        st.error("Model yüklenirken bir hata oluştu.")
        return

    # Sidebar
    st.sidebar.header("Ayarlar")
    threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
    
    tab1, tab2 = st.tabs(["Tekli Görsel Analizi", "Batch (Klasör) İşleme"])

    # Tab 1: Single Image
    with tab1:
        st.subheader("Görsel Yükle veya Yol Belirt")
        
        input_type = st.radio("Girdi Yöntemi:", ["Dosya Yükle", "Dosya Yolu Gir"])
        image = None

        if input_type == "Dosya Yükle":
            uploaded_file = st.file_uploader("Bir görsel seçin...", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
        else:
            path_input = st.text_input("Görselin tam yolunu girin:")
            if path_input and os.path.exists(path_input):
                try:
                    image = Image.open(path_input).convert("RGB")
                except:
                    st.error("Dosya yolu geçersiz veya bir görsel değil.")

        if image:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Yüklenen Görsel", use_container_width=True)
            
            with col2:
                st.write("### Analiz Sonuçları")
                predictions = predict_image(image, processor, model, device, threshold)
                
                if not predictions:
                    st.warning("⚠️ Model emin değil (Tüm sonuçlar threshold altında).")
                else:
                    # Results Table
                    results_df = pd.DataFrame(predictions)
                    results_df['confidence'] = results_df['confidence'].apply(lambda x: f"{x:.2%}")
                    st.table(results_df)

                    # Visualization
                    st.write("#### Confidence Dağılımı")
                    labels = [p['label'] for p in predictions]
                    scores = [p['confidence'] for p in predictions]
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    colors = plt.cm.viridis(scores)
                    bars = ax.barh(labels, scores, color=colors)
                    ax.invert_yaxis()
                    ax.set_xlabel('Confidence')
                    ax.set_title('Top-5 Tahminler')
                    st.pyplot(fig)

    # Tab 2: Batch Processing
    with tab2:
        st.subheader("Klasördeki Tüm Görselleri Analiz Et")
        dir_path = st.text_input("İşlenecek klasörün yolunu girin:")
        
        if st.button("Analizi Başlat"):
            if dir_path:
                with st.spinner("Klasör işleniyor..."):
                    df, error = process_directory(dir_path, processor, model, device, threshold)
                    
                    if error:
                        st.error(error)
                    else:
                        st.success(f"Başarıyla tamamlandı! {len(df)} görsel analiz edildi.")
                        st.dataframe(df, use_container_width=True)
                        
                        csv_path = save_to_csv(df)
                        if csv_path:
                            with open(csv_path, "rb") as f:
                                st.download_button(
                                    label="Sonuçları CSV Olarak İndir",
                                    data=f,
                                    file_name="batch_results.csv",
                                    mime="text/csv"
                                )
            else:
                st.warning("Lütfen bir klasör yolu girin.")

if __name__ == "__main__":
    main()
