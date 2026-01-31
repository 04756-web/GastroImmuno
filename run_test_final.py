import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from rdkit import Chem
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear

# ==========================================
# ⚙️ ตั้งค่าไฟล์ (ผมแก้ให้ตรงกับไฟล์แม่แล้ว)
# ==========================================
YOUR_CSV_FILENAME = 'dataset_PD1_PDL1_FINAL.csv'  
SMILES_COLUMN = 'smiles'   # ชื่อหัวตารางเล็ก
TARGET_COLUMN = 'label'    # ชื่อหัวตารางเล็ก

# ==========================================
# 🧠 โครงสร้างโมเดล (GNN)
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

def smile_to_graph_data(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None: return None, None, None
    atom_features = [[a.GetAtomicNum(), a.GetExplicitValence(), int(a.GetIsAromatic())] for a in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    edge_index += [[e[1], e[0]] for e in edge_index]
    if not edge_index: edge_index = [[],[]]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    batch = torch.zeros(x.size(0), dtype=torch.long)
    return x, edge_index, batch

# ==========================================
# 🚀 เริ่มการตรวจข้อสอบ
# ==========================================
if __name__ == "__main__":
    print(f"📂 กำลังอ่านไฟล์: {YOUR_CSV_FILENAME} ...")
    
    # 1. โหลดข้อมูล
    try:
        df = pd.read_csv(YOUR_CSV_FILENAME)
        print(f"✅ พบข้อมูลทั้งหมด: {len(df)} แถว")
    except FileNotFoundError:
        print("❌ Error: ไม่พบไฟล์ CSV (เช็คว่าไฟล์ dataset_PD1_PDL1_FINAL.csv อยู่ที่เดียวกับโค้ดไหม)")
        exit()

    # 2. โหลดโมเดล
    device = torch.device('cpu')
    model = GNN()
    try:
        model.load_state_dict(torch.load('pd1_best_model.pth', map_location=device))
        model.eval()
        print("✅ โหลดสมอง AI (Model) เรียบร้อย")
    except FileNotFoundError:
        print("❌ Error: ไม่พบไฟล์โมเดล 'pd1_best_model.pth' (ต้องเทรนก่อนนะแม่)")
        exit()

    # 3. แบ่งสอบ (ใช้ 20% เหมือนตอนเทรนเป๊ะๆ เพื่อความแฟร์)
    # เราต้องใช้ random_state เดิมเพื่อให้ได้ชุดข้อสอบชุดเดิม
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"🧪 เริ่มทดสอบกับข้อมูล Test Set จำนวน {len(test_df)} ข้อ...")

    y_true = []
    y_pred = []
    y_probs = []

    # วนลูปตรวจทีละข้อ
    correct = 0
    total = 0
    
    print("   กำลังประมวลผล", end="")
    for index, row in test_df.iterrows():
        smiles = row[SMILES_COLUMN]
        true_label = row[TARGET_COLUMN]
        
        x, edge_index, batch = smile_to_graph_data(smiles)
        
        if x is not None:
            with torch.no_grad():
                out = model(x, edge_index, batch)
                prob = torch.sigmoid(out).item()
                prediction = 1 if prob >= 0.5 else 0
            
            y_true.append(true_label)
            y_probs.append(prob)
            y_pred.append(prediction)
            
            if prediction == true_label:
                correct += 1
            total += 1
            
            if total % 10 == 0:
                print(".", end="", flush=True)

    print("\n✅ ตรวจเสร็จสิ้น!")

    # 4. คำนวณคะแนน
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_probs)
    except:
        auc = 0.5 # กรณีข้อมูลน้อยเกินไป

    cm = confusion_matrix(y_true, y_pred)

    # 5. แสดงผลลัพธ์
    print("\n" + "="*40)
    print("📊 ผลคะแนนจริงของโมเดล (จดเลขนี้ไปใช้นะครับ)")
    print("="*40)
    print(f"🎯 Accuracy (ความแม่นยำ):   {acc:.4f}  ({acc*100:.2f}%)")
    print(f"✨ Precision (ความชัดเจน):  {prec:.4f}  ({prec*100:.2f}%)")
    print(f"🔎 Recall (ความครบถ้วน):    {rec:.4f}  ({rec*100:.2f}%)")
    print(f"⚖️ F1-Score (คะแนนเฉลี่ย):   {f1:.4f}")
    print(f"📈 ROC-AUC Score:          {auc:.4f}")
    print("-" * 20)
    print("Confusion Matrix (ตารางผลลัพธ์):")
    print(cm)
    print(f"(ทายถูก {correct} ข้อ จากทั้งหมด {total} ข้อ)")
    print("="*40)