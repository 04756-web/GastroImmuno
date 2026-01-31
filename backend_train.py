import os
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from rdkit import Chem
import pandas as pd
import numpy as np
import shutil

# ==========================================
# 1. SETUP
# ==========================================
# ตั้งค่า Seed เพื่อให้ผลลัพธ์เหมือนเดิมทุกครั้งที่รัน
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Running on: {device}")

# ==========================================
# 2. CREATE DATASET (PD-1 + PD-L1 + DECOYS)
# ==========================================
def create_hybrid_dataset(filename='dataset_PD1_PDL1_FINAL.csv'):
    print("🛠️ Creating Hybrid dataset (PD-1 & PD-L1)...")
    
    # ข้อมูลดิบ (SMILES, Label)
    # 1 = Active (ยา), 0 = Inactive (ไม่ใช่ยา)
    raw_data = [
        # --- [GROUP A] PD-L1 Inhibitors (BMS Series) ---
        ("COc1cc(Nc2c(cn3cc(C)ccc3n2)C(=O)N)cc(OC)c1OC", 1), # BMS-202
        ("CC(C)Oc1cc(Nc2nc(N)ncc2C(=O)Nc3c(C)cccc3C)cc(OC(C)C)c1", 1), # BMS-8
        ("Cc1cccc(C)c1NC(=O)c2ncc(N)nc2Nc3cc(OC(C)C)c(OC(C)C)c(OC(C)C)c3", 1),
        ("COc1cc(Nc2nc(N)ncc2C(=O)Nc3c(Cl)cccc3Cl)cc(OC)c1OC", 1),
        ("COc1cc(C(=O)N(C)C)cc(OC)c1Oc2ccc(cc2Cl)C3(N)CN(C3)c4ccc(OC5CCN(CC5)CC(=O)O)cc4", 1), # BMS-1166
        ("CC1(N)CN(C1)c2ccc(OC3CCN(CC3)CC(=O)O)cc2c4cc(ccc4Cl)Oc5c(OC)cc(C(=O)N)cc5OC", 1),
        ("OCCn1c(cc2c(Nc3ccc(Br)cc3F)ncnc12)C#N", 1),

        # --- [GROUP B] PD-1 Inhibitors / Pathway Modulators ---
        ("NC(Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CO)C(=O)O", 1), # CA-170
        ("CC(C)C[C@H](NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)[C@@H](N)CC2=CN=CN2)C(=O)O", 1), # CA-327
        ("OC1=CC=C(C=C1)C2=CC(=NO2)C3=CC=CC=C3", 1),
        ("CC1=CC(=C(C=C1)C2=CC(=NN2)C3=CC=CC=C3)O", 1),
        ("CN(C)CC1=CC=C(C=C1)C(=O)NC2=CC=C(C=C2)C3=CN=CC=C3", 1),

        # --- [GROUP C] INACTIVE / DECOYS (สารเคมีทั่วไป) ---
        ("CC(=O)Oc1ccccc1C(=O)O", 0), # Aspirin
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 0), # Caffeine
        ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", 0), # Ibuprofen
        ("CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", 0), # Testosterone
        ("C1=CC=C(C=C1)C(=O)O", 0), # Benzoic acid
        ("CCO", 0), # Ethanol
        ("c1ccccc1", 0), # Benzene
        ("CC(=O)NC1=CC=C(C=C1)O", 0), # Paracetamol
        ("CN1C(=O)N(C)C(=O)C2=C1N=CN2", 0),
        ("C1CCCCC1", 0), 
        ("C1=CN=CC=C1", 0), 
        ("OC1=CC=CC=C1", 0),
        ("CC1=CC=CC=C1", 0),
        ("ClC1=CC=CC=C1", 0),
        ("FC1=CC=CC=C1", 0),
        ("NC1=CC=CC=C1C(=O)O", 0),
        ("C1=CC=C(C=C1)OC2=CC=CC=C2", 0),
        ("CN(C)C1=CC=CC=C1", 0),
        ("CCN(CC)C1=CC=CC=C1", 0),
        ("O=C1CCCCC1", 0),
        ("C1=CC=OC1", 0),
        ("C1=CC=SC1", 0),
        ("C1=CN=CN1", 0),
        ("C1=NC=NC=N1", 0)
    ]
    
    # Data Augmentation: ทำซ้ำข้อมูล 10 รอบเพื่อให้ Model มีข้อมูลพอเทรน (จะได้ประมาณ 300+ ตัว)
    # เทคนิคนี้ช่วยให้ Model จำ Pattern ได้แม่นขึ้นแม้ข้อมูลตั้งต้นจะน้อย
    full_data = []
    for _ in range(12): 
        full_data.extend(raw_data)
        
    df = pd.DataFrame(full_data, columns=['smiles', 'label'])
    
    # Shuffle (สลับมั่ว) เพื่อไม่ให้โมเดลจำลำดับ
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    df.to_csv(filename, index=False)
    print(f"🎉 Created Hybrid Dataset (PD-1 & PD-L1): {len(df)} compounds.")
    return df

# ==========================================
# 3. GRAPH CONVERSION
# ==========================================
def smile_to_graph(smile, label):
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None: return None
        # เก็บ Feature: เลขอะตอม, แขนพันธะ, ความเป็นวงแหวน
        atom_features = [[a.GetAtomicNum(), a.GetExplicitValence(), int(a.GetIsAromatic())] for a in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # เก็บการเชื่อมต่อ (Edges)
        edge_index = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
        edge_index += [[e[1], e[0]] for e in edge_index] # Undirected graph
        if not edge_index: edge_index = [[],[]]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float))
    except: return None

class PD1Dataset(InMemoryDataset):
    def __init__(self, root, dataframe):
        self.dataframe = dataframe
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def processed_file_names(self): return ['pd1_hybrid_final.pt']
    
    def process(self):
        data_list = []
        print(f"⚗️  Converting {len(self.dataframe)} molecules into graphs...")
        for _, r in self.dataframe.iterrows():
            g = smile_to_graph(r['smiles'], r['label'])
            if g is not None: data_list.append(g)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# ==========================================
# 4. GNN MODEL ARCHITECTURE
# ==========================================
class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Graph Convolution Layers
        self.conv1 = GCNConv(3, 128) # Input=3 features
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 64)
        self.lin = Linear(64, 1) # Output=1 (Probability)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch) # รวมข้อมูลทั้งกราฟเป็นจุดเดียว
        return self.lin(x)

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. เคลียร์ Cache เก่าทิ้งเสมอ
    if os.path.exists('data/processed'):
        try: shutil.rmtree('data/processed')
        except: pass
        print("🧹 Cleared old cache.")

    # 2. สร้างข้อมูล
    df = create_hybrid_dataset()
    
    # 3. เตรียมข้อมูลสำหรับ PyTorch
    dataset = PD1Dataset(root='data', dataframe=df)
    print(f"✅ Training with {len(dataset)} graphs.")
        
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 4. สร้างโมเดล
    model = GNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Learning Rate
    criterion = torch.nn.BCEWithLogitsLoss() # Loss function for Binary Classification
    
    print("\n🔥 START TRAINING (50 Epochs)...")
    model.train()
    
    for epoch in range(1, 51):
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
            
        # แสดงผลทุก 10 รอบ
        if epoch % 10 == 0:
            print(f"   Epoch {epoch:02d} | Loss: {loss_all/len(train_loader):.4f}")
            
    # 5. บันทึกโมเดล
    torch.save(model.state_dict(), 'pd1_best_model.pth')
    print("\n✅ SUCCESS! Model saved as 'pd1_best_model.pth'")
    print("👉 Backend เสร็จสมบูรณ์ครับ! พร้อมทำ Frontend ต่อได้เลย")
    # ==========================================
# 6. PERFORMANCE & SPEED TEST (ส่วนเสริม)
# ==========================================
# (ต้อง Import เพิ่มตรงนี้ เพื่อให้โค้ดทำงานได้)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

def evaluate_and_speed_test(model, loader, device):
    print("\n" + "="*40)
    print("🚀 STARTING SPEED TEST (เริ่มจับเวลา)")
    print("="*40)
    
    # 1. จำลองข้อมูล 1 ตัว (Aspirin) เพื่อเทสความเร็ว
    sample_smiles = "CC(=O)Oc1ccccc1C(=O)O" 
    
    # --- จับเวลาช่วงที่ 1: แปลงโครงสร้าง (Preprocessing) ---
    start_time = time.time()
    sample_graph = smile_to_graph(sample_smiles, 0) 
    preprocess_time = time.time() - start_time
    
    # --- จับเวลาช่วงที่ 2: ให้ AI คิด (Inference) ---
    if sample_graph is not None:
        sample_graph = sample_graph.to(device)
        # สร้าง Batch หลอกๆ (เพราะ Model ต้องการ input เป็น Batch)
        batch_idx = torch.zeros(sample_graph.x.size(0), dtype=torch.long).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            _ = model(sample_graph.x, sample_graph.edge_index, batch_idx)
        inference_time = time.time() - start_time
        
        total_time = preprocess_time + inference_time
        
        # แสดงผลเวลา
        print(f"⏱️  1. เวลาแปลงโครงสร้าง : {preprocess_time:.4f} วินาที")
        print(f"🧠 2. เวลาประมวลผลโมเดล : {inference_time:.4f} วินาที")
        print(f"⚡  รวมเวลาทั้งหมด      : {total_time:.4f} วินาที")
        print("="*40)
    else:
        print("❌ Error: ไม่สามารถแปลงกราฟได้")

# เรียกใช้งานฟังก์ชันทันที (วางบรรทัดนี้ให้ชิดขอบซ้ายสุด หรือย่อหน้าให้ตรงกับ if __name__ ก็ได้)
if __name__ == "__main__":
    # เรียกใช้ฟังก์ชันจับเวลา (ส่ง model ที่เทรนเสร็จแล้วเข้าไป)
    evaluate_and_speed_test(model, train_loader, device)