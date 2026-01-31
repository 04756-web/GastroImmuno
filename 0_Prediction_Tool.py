import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem
from fpdf import FPDF
import base64
from datetime import datetime
import time  # Library จับเวลา

# --- Library สำหรับ 3D ---
from stmol import showmol
import py3Dmol

# ==========================================
# 1. ตั้งค่าหน้าเว็บ
# ==========================================
APP_NAME = "GastroImmuno AI"
SUB_TITLE = "PD-1/PD-L1 Screening for Gastric Cancer Immunotherapy"
VERSION = "v1.0.0 (Official Release)"

st.set_page_config(page_title=APP_NAME, page_icon="🦀", layout="wide")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# Header
c1, c2 = st.columns([0.8, 0.2])
with c1:
    st.title(f"🧬 {APP_NAME}")
    st.caption(f"{SUB_TITLE}")
with c2:
    st.markdown(f"**{VERSION}**")

# ==========================================
# 🧠 2. ส่วน Deep Learning Model (GNN)
# ==========================================
class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 64)
        self.lin = Linear(64, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)

@st.cache_resource
def load_model():
    model = GNN()
    try:
        model.load_state_dict(torch.load('pd1_best_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        return None

model = load_model()

def smile_to_graph_data(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None: return None
    atom_features = [[a.GetAtomicNum(), a.GetExplicitValence(), int(a.GetIsAromatic())] for a in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    edge_index += [[e[1], e[0]] for e in edge_index]
    if not edge_index: edge_index = [[],[]]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    batch = torch.zeros(x.size(0), dtype=torch.long)
    return x, edge_index, batch

# ==========================================
# 🧪 ฟังก์ชันสร้างโมเดล 3D
# ==========================================
def make_3d_view(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol = Chem.AddHs(mol) 
        AllChem.EmbedMolecule(mol, randomSeed=42) 
        mblock = Chem.MolToMolBlock(mol)
        
        view = py3Dmol.view(width=500, height=400)
        view.addModel(mblock, 'mol')
        view.setStyle({'stick': {}})
        view.setBackgroundColor('#FFFFFF')
        view.zoomTo()
        return view
    return None

# ==========================================
# 📄 3. ฟังก์ชันสร้าง PDF Report
# ==========================================
def create_pdf(smiles, label, confidence, mol_wt, logp):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(0, 51, 102)
    pdf.rect(0, 0, 210, 25, 'F')
    pdf.set_y(30)
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 10, f"{APP_NAME}: Analysis Report", ln=True, align='C')
    pdf.set_font("Arial", 'I', 11)
    pdf.cell(0, 5, "Immunotherapy Screening for Gastric Cancer (PD-1/PD-L1)", ln=True, align='C')
    pdf.ln(10)
    
    report_id = f"GC-{int(datetime.now().timestamp())}"
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    pdf.set_font("Courier", '', 10)
    pdf.cell(0, 5, f"Date: {date_str}", ln=True, align='R')
    pdf.cell(0, 5, f"Sample ID: {report_id}", ln=True, align='R')
    pdf.line(10, 60, 200, 60)
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "SCREENING RESULT", ln=True)
    pdf.ln(2)
    pdf.set_font("Arial", 'B', 12)
    if "ACTIVE" in label:
        pdf.set_text_color(0, 100, 0)
    else:
        pdf.set_text_color(180, 0, 0)
    pdf.cell(50, 10, "Activity Status:", border=1)
    pdf.cell(0, 10, f"  {label}", border=1, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(50, 10, "Target Probability:", border=1)
    pdf.cell(0, 10, f"  {confidence}", border=1, ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "COMPOUND PROPERTIES", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8, f"SMILES Structure:\n{smiles}")
    pdf.ln(5)
    pdf.cell(60, 8, f"Molecular Weight: {mol_wt} g/mol", border=0)
    pdf.cell(60, 8, f"LogP (Lipophilicity): {logp}", border=0, ln=True)
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 🖥️ 4. หน้าจอ UI หลัก
# ==========================================
if 'analyzed' not in st.session_state:
    st.session_state['analyzed'] = False
    st.session_state['result_data'] = {}

col1, col2 = st.columns([1, 1])

# --- Column 1: Input & Processing ---
with col1:
    st.info("กรอกโครงสร้างโมเลกุล (SMILES) เพื่อทดสอบ")
    default_smiles = "COc1cc(Nc2c(cn3cc(C)ccc3n2)C(=O)N)cc(OC)c1OC"
    smiles_input = st.text_area("Input SMILES:", value=default_smiles, height=100)
    
    analyze_btn = st.button("🚀 Run Analysis (Full System)", type="primary")

    if analyze_btn:
        if model is None:
            st.error("❌ ไม่พบไฟล์โมเดล (pd1_best_model.pth)")
        else:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                # ---------------------------------------------------------
                # ⏱️ เริ่มจับเวลา: รวมตั้งแต่แปลงข้อมูล ยันสร้างภาพ 3D
                # ---------------------------------------------------------
                
                # 1. จับเวลาแปลงข้อมูล (Conversion)
                t1 = time.time()
                x, edge_index, batch = smile_to_graph_data(smiles_input)
                t2 = time.time()
                time_conv = t2 - t1
                
                # 2. จับเวลา AI ทำนาย (Inference)
                with torch.no_grad():
                    logits = model(x, edge_index, batch)
                    prob = torch.sigmoid(logits).item()
                t3 = time.time()
                time_ai = t3 - t2
                
                # 3. จับเวลาสร้างภาพ 3D (Visualization Prep) -> ส่วนนี้แหละที่ทำให้นานขึ้นจริง
                view_obj = make_3d_view(smiles_input)
                mw = Descriptors.MolWt(mol) # คำนวณแถมไปด้วย
                logp = Descriptors.MolLogP(mol)
                t4 = time.time()
                time_vis = t4 - t3
                
                # 4. เวลารวมทั้งหมดที่ User ต้องรอ
                time_total = t4 - t1

                # ---------------------------------------------------------
                # 🖨️ ปริ้นท์ผลลัพธ์ลงจอดำ (Terminal)
                # ---------------------------------------------------------
                print(f"\n{'='*50}")
                print(f"⏱️ REAL USER WAITING TIME (เวลารวมระบบ)")
                print(f"{'-'*50}")
                print(f"1. แปลงโครงสร้าง (Conversion) : {time_conv:.6f} s")
                print(f"2. AI ประมวลผล (Inference)    : {time_ai:.6f} s")
                print(f"3. สร้างภาพ 3 มิติ (3D Render) : {time_vis:.6f} s  <-- นานสุดตรงนี้")
                print(f"{'-'*50}")
                print(f"🚀 เวลารวมทั้งหมด (Total Time)  : {time_total:.6f} s")
                print(f"{'='*50}\n")
                
                # บันทึกลง Session
                st.session_state['analyzed'] = True
                st.session_state['result_data'] = {
                    'prob': prob,
                    'mw': mw,
                    'logp': logp,
                    'smiles': smiles_input,
                    'view_obj': view_obj # เก็บตัว 3D ที่สร้างเสร็จแล้วไว้
                }
            else:
                st.error("Invalid SMILES format. Please check input.")
                st.session_state['analyzed'] = False

# --- Column 2: Visualization (3D & Results) ---
with col2:
    st.markdown("#### 🧬 Structure Visualization (3D)")
    
    if st.session_state['analyzed']:
        data = st.session_state['result_data']
        
        # แสดงผล 3D (ใช้ตัวที่สร้างไว้แล้ว ไม่ต้องคำนวณใหม่)
        if 'view_obj' in data and data['view_obj'] is not None:
             showmol(data['view_obj'], height=400, width=500)
             st.caption("💡 Tip: Use mouse to rotate / Zoom in-out")
        
        # แสดงค่าตัวเลข (Properties)
        c1_sub, c2_sub = st.columns(2)
        c1_sub.metric("Molecular Weight", f"{data['mw']:.2f}")
        c2_sub.metric("LogP", f"{data['logp']:.2f}")
        
        st.markdown("---")
        
        # แสดงผลทำนาย Active/Inactive
        percentage = data['prob'] * 100
        if data['prob'] >= 0.5:
            label = "ACTIVE (Inhibitor)"
            color = "#28a745" # Green
            icon = "✅"
        else:
            label = "INACTIVE"
            color = "#dc3545" # Red
            icon = "❌"

        st.markdown(f"#### Screening Result")
        st.markdown(f"<div style='text-align: center; color: {color}; border: 2px solid {color}; padding: 10px; border-radius: 10px;'><h3>{icon} {label}</h3></div>", unsafe_allow_html=True)
        st.progress(data['prob'])
        st.caption(f"Confidence Score: {percentage:.2f}%")
        
        # ปุ่มโหลด PDF
        pdf_bytes = create_pdf(data['smiles'], label, f"{percentage:.2f}%", f"{data['mw']:.2f}", f"{data['logp']:.2f}")
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f"""
        <a href="data:application/octet-stream;base64,{b64}" download="GastroImmuno_Report.pdf" 
           style="text-decoration:none; color:black; background-color:#f0f2f6; padding:10px; 
                  border-radius:5px; border:1px solid #ccc; display:block; text-align:center; margin-top:10px;">
           📄 Download Report
        </a>
        """
        st.markdown(href, unsafe_allow_html=True)
        
    else:
        st.info("Waiting for analysis...")
        st.markdown(
            """
            <div style="
                border: 2px dashed #ccc; 
                border-radius: 10px; 
                height: 400px; 
                display: flex; 
                align-items: center; 
                justify_content: center; 
                background-color: #f9f9f9;
                color: #888;">
                <h3> 3D Model will appear here</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )