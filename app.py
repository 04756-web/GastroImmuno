import streamlit as st
import base64
import os

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="BioScreen AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. ฟังก์ชันโหลดภาพและ CSS
def set_background(image_file):
    # เช็คไฟล์
    if not os.path.exists(image_file):
        st.error(f"⚠️ หาไฟล์ไม่เจอ: {image_file}")
        return

    with open(image_file, "rb") as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    
    css = f"""
    <style>
    /* 1. จัดการพื้นหลัง */
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* 2. ซ่อน Header และ Footer เดิมของ Streamlit */
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* 3. ล็อคตำแหน่งปุ่มให้อยู่ตรงกลางล่าง เป๊ะๆ! */
    div.stButton {{
        position: fixed;        /* ล็อคตำแหน่งไว้กับหน้าจอ */
        bottom: 100px;          /* ห่างจากขอบล่าง 100px (ปรับเลขนี้เพื่อขึ้น-ลง) */
        left: 50%;              /* จุดเริ่มอยู่ที่ 50% ของหน้าจอ */
        transform: translateX(-50%); /* ขยับกลับมาครึ่งปุ่ม เพื่อให้กลางเป๊ะ */
        z-index: 9999;          /* ให้ลอยอยู่เหนือทุกอย่าง */
        text-align: center;
        width: auto;
    }}
    
    /* 4. ปรับหน้าตาปุ่ม */
    div.stButton > button {{
        background-color: rgba(0, 201, 255, 0.25); /* โปร่งแสงนิดๆ */
        color: white;
        font-size: 26px;        /* ตัวหนังสือใหญ่ขึ้น */
        font-weight: bold;
        padding: 15px 50px;     /* ปรับขนาดปุ่ม (บนล่าง ซ้ายขวา) */
        border-radius: 50px;
        border: 2px solid #00C9FF;
        box-shadow: 0 0 20px rgba(0, 201, 255, 0.6);
        backdrop-filter: blur(8px);
        transition: all 0.3s ease-in-out;
    }}
    
    div.stButton > button:hover {{
        background-color: #00C9FF;
        color: white;
        box-shadow: 0 0 50px rgba(0, 201, 255, 1);
        transform: scale(1.05);
    }}

    /* 5. Footer ที่สร้างเอง ล็อคไว้ล่างสุด */
    .custom-footer {{
        position: fixed;
        bottom: 10px;
        left: 0;
        width: 100%;
        text-align: center;
        color: rgba(255,255,255,0.6);
        font-size: 12px;
        z-index: 9998;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- เริ่มการทำงาน ---
set_background('assets/background.jpg')

# ไม่ต้องใช้ st.columns หรือ st.write("") ดันบรรทัดแล้ว
# วางปุ่มลงไปเลย CSS ข้างบนจะจับมันไปล็อคที่เดิมเอง
if st.button("CLICK TO ENTER SYSTEM | เข้าสู่ระบบ"):
    st.switch_page("pages/0_Prediction_Tool.py")

# Footer
st.markdown('<div class="custom-footer">GastroImmuno | Developed by PCSHSL</div>', unsafe_allow_html=True)