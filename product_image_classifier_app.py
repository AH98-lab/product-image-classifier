
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# تحميل النموذج
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("keras_model.h5")
    return model

model = load_model()

# أسماء الفئات
class_names = ['Clothing', 'Furniture', 'Home Tools', 'Electronics', 'Food']

# واجهة التطبيق
st.set_page_config(page_title="تصنيف صور المنتجات", layout="centered")
st.title("🛍️ أداة تصنيف صور المنتجات بالذكاء الاصطناعي")
st.write("قم بتحميل صورة منتج، وسنخبرك إلى أي فئة تنتمي:")

uploaded_file = st.file_uploader("📤 حمّل صورة المنتج", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='📷 الصورة التي قمت برفعها', use_column_width=True)

    image = image.resize((224, 224))
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    predictions = model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.success(f"✅ التصنيف: **{predicted_class}**")
    st.bar_chart(predictions[0])
