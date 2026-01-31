import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Knowledge Base", page_icon="📘", layout="wide")

# --- 🧭 NAVIGATION BAR ---
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
nav1, nav2, nav3, nav4 = st.columns(4)
with nav1: st.page_link("pages/0_Prediction_Tool.py", label="Prediction", icon=":material/science:", use_container_width=True)
with nav2: st.page_link("pages/1_📘_Knowledge_Base.py", label="Knowledge", icon=":material/menu_book:", use_container_width=True)
with nav3: st.page_link("pages/2_Model_Performance.py", label="Performance", icon=":material/bar_chart:", use_container_width=True)
with nav4: st.page_link("pages/3_About_Us.py", label="About", icon=":material/info:", use_container_width=True)
st.markdown("---")

# 2. CSS แต่งหน้าตา
st.markdown("""
<style>
    .info-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #eee;
    }
    h3 { color: #2c3e50; }
    p { color: #555; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

st.title("📘 PD-1/PD-L1 Inhibitor Knowledge Base")
st.write("คลังความรู้เกี่ยวกับกลไกการยับยั้ง Checkpoint และการออกแบบยา (Drug Discovery)")

# --- แบ่งเนื้อหาเป็น Tabs ---
tab1, tab2, tab3 = st.tabs(["🧬 Biological Mechanism", "💊 Small Molecule Inhibitors", "📚 Case Studies"])

# --- TAB 1: กลไกทางชีวภาพ ---
with tab1:
    st.markdown("### The PD-1/PD-L1 Pathway")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>🛡️ ระบบภูมิคุ้มกันทำงานอย่างไร?</h4>
            <p>
                <b>PD-1 (Programmed Cell Death Protein 1)</b> เป็นโปรตีนที่อยู่บนผิวของเซลล์เม็ดเลือดขาว (T-cells) 
                ทำหน้าที่เป็น "เบรก" เพื่อป้องกันไม่ให้ภูมิคุ้มกันทำลายเซลล์ดีในร่างกาย
            </p>
            <p>
                แต่ทว่า... <b>เซลล์มะเร็ง (Tumor Cell)</b> ฉลาดแกมโกง มันสร้างโปรตีนชื่อ <b>PD-L1</b> 
                มาจับกับ PD-1 ของ T-cell ทำให้ T-cell เข้าใจผิดคิดว่าเป็นพวกเดียวกัน และ "หยุดทำงาน" 
                ส่งผลให้มะเร็งเติบโตต่อไปได้
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 **Key Concept:** การยับยั้ง (Inhibition) คือการเอายาไปขวางไม่ให้ PD-1 จับกับ PD-L1 ได้ ทำให้ T-cell กลับมาฆ่ามะเร็งได้อีกครั้ง")

    with col2:
        # ใช้รูป Diagram มาตรฐานจาก Wikimedia (ถ้าแม่มีรูปเอง เปลี่ยน Link หรือใช้ st.image("assets/my_pic.jpg") ได้เลย)
        st.image("assets/PD.jpg", 
                 caption="กลไกการทำงานของ PD-1/PD-L1", use_container_width=True)

# --- TAB 2: สารยับยั้งโมเลกุลเล็ก ---
with tab2:
    st.markdown("### Why Small Molecules?")
    
    # 3 จุดเด่น (ใช้ Columns)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.container(border=True).markdown("""
        #### 💰 Cost Effective
        มีต้นทุนการผลิตที่ต่ำกว่ายาประเภท Antibodies (Biologics) มาก ทำให้ผู้ป่วยเข้าถึงง่ายขึ้น
        """)
    with c2:
        st.container(border=True).markdown("""
        #### 💊 Oral Bioavailability
        สามารถทำเป็น "ยาเม็ด" ทานได้ ไม่ต้องฉีดเข้าเส้นเลือด เหมือนยา Antibodies ทั่วไป
        """)
    with c3:
        st.container(border=True).markdown("""
        #### 🎯 Tumor Penetration
        ด้วยขนาดที่เล็ก ทำให้แทรกซึมเข้าสู่ก้อนเนื้อเยื่อมะเร็ง (Solid Tumors) ได้ดีกว่า
        """)

    st.markdown("---")
    
    # ส่วนแสดงโครงสร้างเคมี (ใช้ Code วาดสดๆ ให้ดูเยอะ)
    st.markdown("### Structural Classes of Inhibitors")
    st.write("ตัวอย่างโครงสร้างทางเคมีที่ถูกค้นพบว่ามีฤทธิ์ยับยั้ง PD-1/PD-L1")

    # ฟังก์ชันวาดรูปเคมี
    def show_chemical(smiles, name):
        mol = Chem.MolFromSmiles(smiles)
        img = Draw.MolToImage(mol, size=(300, 200))
        st.image(img, caption=name)

    row1_c1, row1_c2, row1_c3 = st.columns(3)
    
    with row1_c1:
        st.markdown("**1. BMS-202 (Active)**")
        show_chemical("COc1cc(CNCC2(CCCC2)NCc3cccc(-c4ccccc4)c3)cc(OC)c1", "BMS-202 Structure")
        st.caption("สารต้นแบบที่มีค่า IC50 ต่ำ (ยับยั้งได้ดี)")

    with row1_c2:
        st.markdown("**2. BMS-1166 (Potent)**")
        show_chemical("COc1c(Cl)cc(CNCC2(CCN(CC2)C(=O)C(O)(C)C)C)cc1OCc3cccc(-c4ccc(cn4)C#N)c3", "BMS-1166 Structure")
        st.caption("พัฒนาต่อยอด เพิ่มหมู่ Chlorine ช่วยในการจับตัว")

    with row1_c3:
        st.markdown("**3. Inactive Control**")
        show_chemical("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin (Negative Control)")
        st.caption("โครงสร้างที่ไม่จับกับ PD-1 (ใช้เปรียบเทียบ)")

# --- TAB 3: กรณีศึกษาและงานวิจัย ---
with tab3:
    st.markdown("### 📚 Research & Development")
    
    with st.expander("📌 BMS Series (Bristol-Myers Squibb)", expanded=True):
        st.write("""
        บริษัท BMS เป็นผู้บุกเบิกการจดสิทธิบัตรสารประกอบโมเลกุลเล็กที่ยับยั้ง PD-1/PD-L1 
        โดยใช้โครงสร้างพื้นฐานแบบ **Biphenyl Core** ซึ่งเป็นต้นแบบที่โมเดล AI ของเราใช้เรียนรู้ 
        สารในกลุ่มนี้ทำงานโดยการเหนี่ยวนำให้โปรตีน PD-L1 เกิดการรวมตัวกัน (Dimerization) จนไม่สามารถไปจับกับ PD-1 ได้
        """)
    
    with st.expander("📌 Challenges in Drug Design"):
        st.write("""
        ความยากของการออกแบบยานี้คือ **Protein-Protein Interaction (PPI)** เพราะพื้นผิวรอยต่อระหว่าง PD-1 และ PD-L1 นั้นกว้างและแบน (Flat surface) 
        ทำให้หาโมเลกุลเล็กๆ ไปเกาะได้ยากกว่าเอนไซม์ทั่วไป AI จึงมีบทบาทสำคัญในการคัดกรองสารนับล้านตัว
        """)
    
    st.markdown("---")
    st.markdown("#### 🔗 References")
    st.markdown("""
    <div style="font-size: 13px; color: #666;">
    1. Zak, K. M., et al. (2016). Structural basis for small molecule targeting of the programmed death ligand 1 (PD-L1). <i>Oncotarget</i>.<br>
    2. Guzik, K., et al. (2017). Small-molecule inhibitors of the programmed cell death-1/programmed death-ligand 1 (PD-1/PD-L1) interaction via transient dimerization. <i>Journal of Medicinal Chemistry</i>.
    </div>
    """, unsafe_allow_html=True)