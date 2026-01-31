import streamlit as st
import base64
import os

st.set_page_config(page_title="About Team", page_icon="ℹ️", layout="wide")

# ==========================================
# 🛠️ ฟังก์ชันช่วยแปลงรูปภาพ (เพื่อให้แสดงผลได้ชัวร์)
# ==========================================
def get_img_as_base64(file_path):
    # เช็คว่ามีไฟล์อยู่จริงไหม
    if not os.path.exists(file_path):
        return "" # ถ้าไม่มีไฟล์ ให้คืนค่าว่าง
    
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- 🧭 NAVIGATION BAR ---
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
nav1, nav2, nav3, nav4 = st.columns(4)
with nav1: st.page_link("pages/0_Prediction_Tool.py", label="Prediction", icon=":material/science:", use_container_width=True)
with nav2: st.page_link("pages/1_📘_Knowledge_Base.py", label="Knowledge", icon=":material/menu_book:", use_container_width=True)
with nav3: st.page_link("pages/2_Model_Performance.py", label="Performance", icon=":material/bar_chart:", use_container_width=True)
with nav4: st.page_link("pages/3_About_Us.py", label="About", icon=":material/info:", use_container_width=True)
st.markdown("---")

# ==========================================
# 🎨 CSS ตกแต่ง (รวมคำสั่งขยายรูปให้เต็มวง)
# ==========================================
st.markdown("""
<style>
    .profile-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #f0f0f0;
        height: 100%;
    }
    .profile-img {
        width: 150px;      /* กว้าง */
        height: 150px;     /* สูง */
        border-radius: 50%; /* ทำเป็นวงกลม */
        border: 4px solid #f8f9fa;
        margin-bottom: 15px;
        
        /* ✅ คำสั่งสำคัญ: ขยายรูปให้เต็มวง (ตัดส่วนเกินทิ้ง ไม่บีบรูป) */
        object-fit: cover; 
        object-position: center top; /* จัดให้โฟกัสที่ช่วงบน (ใบหน้า) */
    }
    h3 { margin: 10px 0 5px 0; font-size: 20px; color: #333; }
    .role { color: #28a745; font-weight: bold; font-size: 14px; margin-bottom: 10px; }
    p { color: #666; font-size: 14px; margin: 0; }
    
    /* เส้นคั่นสวยๆ */
    .divider { margin: 40px 0; border-top: 1px solid #eee; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Development Team</h1>", unsafe_allow_html=True)
st.write("") # เว้นบรรทัด

# ==========================================
# 📥 โหลดรูปภาพเตรียมไว้ (แปลงเป็น Base64)
# ==========================================
# ⚠️ ตรวจสอบชื่อไฟล์ในโฟลเดอร์ assets ให้ตรงเป๊ะๆ นะครับ
img_kanyawee = get_img_as_base64("assets/kanyawee.jpg")
img_mintra   = get_img_as_base64("assets/mintra.jpg")
img_wachi    = get_img_as_base64("assets/wachi.jpg")
img_sunantha = get_img_as_base64("assets/sunantha.jpg")

# ==========================================
# 👨‍🎓 ส่วนที่ 1: นักศึกษาผู้จัดทำ (2 คน)
# ==========================================
st.markdown("### 👨‍🎓 Project Developers")
col1, col2 = st.columns(2)

with col1:
    # นักศึกษาคนที่ 1
    st.markdown(f"""
    <div class="profile-card">
        <img src="data:image/jpg;base64,{img_kanyawee}" class="profile-img">
        <h3>กัญญาวีร์ ทิพย์สูตร</h3>
        <div class="role">Data Analyst</div>
        <p>โรงเรียนวิทยาศาสตร์จุฬาภรณราชวิทยาลัย ลพบุรี</p>
        <p>04749@pccl.ac.th</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # นักศึกษาคนที่ 2
    st.markdown(f"""
    <div class="profile-card">
        <img src="data:image/jpg;base64,{img_mintra}" class="profile-img">
        <h3>มินตรา ปัญญาเดชาโชติ</h3>
        <div class="role">Model Development & Web Application</div>
        <p>โรงเรียนวิทยาศาสตร์จุฬาภรณราชวิทยาลัย ลพบุรี</p>
        <p>04756@pccl.ac.th</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==========================================
# 🏆 ส่วนที่ 2: อาจารย์ที่ปรึกษา (2 ท่าน)
# ==========================================
st.markdown("### 🏆 Project Advisors")
adv1, adv2 = st.columns(2)

with adv1:
    # อาจารย์คนที่ 1
    st.markdown(f"""
    <div class="profile-card">
        <img src="data:image/jpg;base64,{img_wachi}" class="profile-img" style="border-color: #FFD700;">
        <h3>วชิรวิทย์ เอี่ยมวิลัย</h3>
        <div class="role">Main Advisor</div>
        <p>สาขาคอมพิวเตอร์</p>
    </div>
    """, unsafe_allow_html=True)

with adv2:
    # อาจารย์คนที่ 2
    st.markdown(f"""
    <div class="profile-card">
        <img src="data:image/jpg;base64,{img_sunantha}" class="profile-img" style="border-color: #FFD700;">
        <h3>สุนันทา ศิริมงคล</h3>
        <div class="role">Co-Advisor</div>
        <p>สาขาคอมพิวเตอร์</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:#999; font-size:12px;'>GastroImmuno © 2026</div>", unsafe_allow_html=True)