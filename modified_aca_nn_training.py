import numpy as np
import time
import os

# --- Main Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Attempt to import cupy for RBM/DBN pre-training
try:
    import cupy as cp
    print("✅ CuPy (GPU) backend found for RBM/DBN pre-training.")
except ImportError:
    print("⚠️ CuPy not found. RBM/DBN pre-training will be skipped.")
    cp = None
    exit()

# ============================================================================
# 0. Setup and Helper Functions
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

try:
    import torch._dynamo
    TORCH_COMPILE_AVAILABLE = True
    print("✅ PyTorch 2.0+ detected. Using torch.compile() for optimization.")
except (ImportError, AttributeError):
    TORCH_COMPILE_AVAILABLE = False
    print("⚠️ PyTorch version < 2.0 detected. torch.compile() will be skipped.")

def generate_bars_and_stripes(height, width, num_samples):
    data = np.zeros((num_samples, height * width), dtype=np.float32)
    for i in range(num_samples):
        if np.random.rand() > 0.5: # Horizontal
            img = np.zeros((height, width), dtype=np.float32); row_indices = np.random.choice(height, np.random.randint(1, height + 1), replace=False); img[row_indices, :] = 1; data[i, :] = img.flatten()
        else: # Vertical
            img = np.zeros((height, width), dtype=np.float32); col_indices = np.random.choice(width, np.random.randint(1, width + 1), replace=False); img[:, col_indices] = 1; data[i, :] = img.flatten()
    return data

def create_labels_for_bars_and_stripes(data, height, width):
    labels = np.zeros(data.shape[0], dtype=np.float32)
    for i in range(data.shape[0]):
        img = data[i].reshape(height, width)
        if np.sum(np.var(img, axis=1)) > np.sum(np.var(img, axis=0)):
            labels[i] = 0 # Horizontal
        else:
            labels[i] = 1 # Vertical
    return labels

def cupy_sigmoid(x):
    return 1. / (1. + cp.exp(-x))

def cupy_gibbs_sampling(v0, W_gpu, b_gpu, c_gpu, k=10):
    vk = v0
    for _ in range(k):
        phk = cupy_sigmoid(cp.dot(vk, W_gpu) + c_gpu)
        hk = (cp.random.rand(*phk.shape) < phk).astype(np.float32)
        pvk = cupy_sigmoid(cp.dot(hk, W_gpu.T) + b_gpu)
        vk = (cp.random.rand(*pvk.shape) < pvk).astype(np.float32)
    phk_final = cupy_sigmoid(cp.dot(vk, W_gpu) + c_gpu)
    return vk, phk_final

# ============================================================================
# 1. RBM and Solvers
# ============================================================================

class RBM(nn.Module):
    def __init__(self, n_vis, n_hid):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_vis, n_hid, device=DEVICE) * 0.1)
        self.v_bias = nn.Parameter(torch.zeros(n_vis, device=DEVICE))
        self.h_bias = nn.Parameter(torch.zeros(n_hid, device=DEVICE))

    @torch.compile(dynamic=True)
    def energy(self, v, h):
        v_term = torch.einsum('rbv,v->rb', v, self.v_bias)
        h_term = torch.einsum('rbh,h->rb', h, self.h_bias)
        w_term = torch.einsum('rbv,vh,rbh->rb', v, self.W, h)
        return -(v_term + h_term + w_term)

class ParallelTemperingRBMSolver:
    def __init__(self, rbm_model, batch_size, num_replicas=16, T_min=0.1, T_max=5.0):
        self.rbm=rbm_model;self.num_replicas=num_replicas;self.temps=torch.logspace(np.log10(T_min),np.log10(T_max),num_replicas).to(DEVICE);self.reset_states(batch_size)
    def reset_states(self,batch_size):
        n_vis,n_hid=self.rbm.W.shape;self.v_replicas=(torch.rand(self.num_replicas,batch_size,n_vis,device=DEVICE)>0.5).float();self.h_replicas=(torch.rand(self.num_replicas,batch_size,n_hid,device=DEVICE)>0.5).float()
    def solve(self,iterations=50):
        with torch.no_grad():
            for _ in range(iterations):
                h_prob=torch.sigmoid(torch.einsum('rbv,vh->rbh',self.v_replicas,self.rbm.W)+self.rbm.h_bias);self.h_replicas=(torch.rand_like(h_prob)<h_prob).float()
                v_prob=torch.sigmoid(torch.einsum('rbh,hv->rbv',self.h_replicas,self.rbm.W.T)+self.rbm.v_bias);self.v_replicas=(torch.rand_like(v_prob)<v_prob).float()
                energies=torch.mean(self.rbm.energy(self.v_replicas,self.h_replicas),dim=1);E1,E2=energies[:-1],energies[1:];beta1,beta2=1.0/self.temps[:-1],1.0/self.temps[1:]
                swap_mask=torch.rand(self.num_replicas-1,device=DEVICE)<torch.exp((beta1-beta2)*(E1-E2))
                v_temp=self.v_replicas[:-1][swap_mask].clone();self.v_replicas[:-1][swap_mask]=self.v_replicas[1:][swap_mask];self.v_replicas[1:][swap_mask]=v_temp
                h_temp=self.h_replicas[:-1][swap_mask].clone();self.h_replicas[:-1][swap_mask]=self.h_replicas[1:][swap_mask];self.h_replicas[1:][swap_mask]=h_temp
        return self.v_replicas[0],self.h_replicas[0]

# ============================================================================
# 2. Training Functions with Hyperparameters
# ============================================================================

def train_rbm_with_gibbs(epochs=100, learning_rate=0.1, hidden_dim=16, k_gibbs_steps=10):
    print("\n" + "="*50); print("--- Training Standalone RBM with CuPy Gibbs Sampling (CD-k) ---"); print("="*50)
    print(f"Hyperparameters: epochs={epochs}, lr={learning_rate}, hidden_dim={hidden_dim}, k={k_gibbs_steps}")
    
    V_dim, H_dim, batch_size = 16, hidden_dim, 32
    data = generate_bars_and_stripes(4, 4, 1000); loader = DataLoader(TensorDataset(torch.from_numpy(data)), batch_size=batch_size, shuffle=True)
    rbm = RBM(V_dim, H_dim).to(DEVICE); optimizer = optim.SGD(rbm.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        epoch_error = 0.0
        for batch_tuple in loader:
            v_pos = batch_tuple[0].to(DEVICE)
            with torch.no_grad():
                v0_cupy=cp.asarray(v_pos.cpu());W_cupy,b_cupy,c_cupy=cp.asarray(rbm.W.cpu()),cp.asarray(rbm.v_bias.cpu()),cp.asarray(rbm.h_bias.cpu())
                v_neg_cupy, _ = cupy_gibbs_sampling(v0_cupy, W_cupy, b_cupy, c_cupy, k=k_gibbs_steps); v_neg = torch.as_tensor(v_neg_cupy, device=DEVICE)
            pos_free_energy = -torch.sum(rbm.v_bias * v_pos, dim=1) - torch.sum(nn.functional.softplus(torch.matmul(v_pos, rbm.W) + rbm.h_bias), dim=1)
            neg_free_energy = -torch.sum(rbm.v_bias * v_neg, dim=1) - torch.sum(nn.functional.softplus(torch.matmul(v_neg, rbm.W) + rbm.h_bias), dim=1)
            loss = (torch.mean(pos_free_energy) - torch.mean(neg_free_energy)); optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_error += torch.mean((v_pos - v_neg)**2).item() * v_pos.size(0)
        if (epoch + 1) % 10 == 0: print(f"Epoch {epoch+1}/{epochs}, Reconstruction Error: {epoch_error / len(data):.4f}")
    print("Standalone RBM training finished.")

def train_dbn(x_train_cpu, y_train_cpu, x_test_cpu, y_test_cpu, hidden_sizes=[256, 128], epochs_pretrain=50, epochs_finetune=30, lr_pretrain=0.05, solver_iterations=25):
    print("\n"+"="*50);print("--- Training DBN with ACA Pre-training and GD Fine-tuning ---");print("="*50)
    print(f"Hyperparameters: hidden_sizes={hidden_sizes}, epochs_pre={epochs_pretrain}, epochs_fine={epochs_finetune}, lr_pre={lr_pretrain}, solver_iter={solver_iterations}")

    batch_size = 32
    print("\n--- PHASE 1: Greedy Layer-wise Pre-training with ACA-like Solver ---")
    current_input_data=torch.from_numpy(x_train_cpu);trained_rbm_layers=[]
    for i,h_size in enumerate(hidden_sizes):
        V_dim=current_input_data.shape[1];print(f"\n--- Pre-training DBN Layer {i+1} (RBM: {V_dim} -> {h_size}) ---")
        rbm=RBM(V_dim,h_size).to(DEVICE);loader=DataLoader(TensorDataset(current_input_data),batch_size=batch_size,shuffle=True);solver=ParallelTemperingRBMSolver(rbm,batch_size=batch_size)
        for epoch in range(epochs_pretrain):
            epoch_error = 0.0
            for batch_tuple in loader:
                v_pos=batch_tuple[0].to(DEVICE)
                if v_pos.size(0)!=solver.v_replicas.size(1):solver.reset_states(v_pos.size(0))
                h_pos_prob=torch.sigmoid(torch.matmul(v_pos,rbm.W)+rbm.h_bias);v_neg,_=solver.solve(iterations=solver_iterations);h_neg_prob=torch.sigmoid(torch.matmul(v_neg,rbm.W)+rbm.h_bias)
                with torch.no_grad():
                    pos_assoc=torch.matmul(v_pos.T,h_pos_prob);neg_assoc=torch.matmul(v_neg.T,h_neg_prob);rbm.W+=lr_pretrain*(pos_assoc-neg_assoc)/v_pos.size(0)
                    rbm.v_bias+=lr_pretrain*torch.mean(v_pos-v_neg,dim=0);rbm.h_bias+=lr_pretrain*torch.mean(h_pos_prob-h_neg_prob,dim=0)
                epoch_error += torch.mean((v_pos - v_neg)**2).item() * v_pos.size(0)
            if(epoch+1)%10==0:print(f"  Layer {i+1}, Pre-train Epoch {epoch+1}/{epochs_pretrain}, Loss: {epoch_error / len(current_input_data):.4f}")
        trained_rbm_layers.append(rbm)
        with torch.no_grad():current_input_data=torch.sigmoid(torch.matmul(current_input_data.to(DEVICE),rbm.W)+rbm.h_bias).cpu()
    print("\nDBN pre-training finished.")
    print("\n--- PHASE 2: Supervised Fine-tuning with Gradient Descent ---")
    dbn_classifier=nn.Sequential();
    for i,rbm_layer in enumerate(trained_rbm_layers):
        linear_layer=nn.Linear(rbm_layer.W.shape[0],rbm_layer.W.shape[1]);linear_layer.weight.data=rbm_layer.W.data.T;linear_layer.bias.data=rbm_layer.h_bias.data
        dbn_classifier.add_module(f"layer_{i}",linear_layer);dbn_classifier.add_module(f"activation_{i}",nn.Sigmoid())
    dbn_classifier.add_module("output_layer",nn.Linear(hidden_sizes[-1],1));dbn_classifier.to(DEVICE);criterion=nn.BCEWithLogitsLoss();optimizer=optim.Adam(dbn_classifier.parameters(),lr=0.001)
    train_dataset=TensorDataset(torch.from_numpy(x_train_cpu),torch.from_numpy(y_train_cpu.reshape(-1,1)));train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_dataset=TensorDataset(torch.from_numpy(x_test_cpu),torch.from_numpy(y_test_cpu.reshape(-1,1)));test_loader=DataLoader(test_dataset,batch_size=batch_size)
    for epoch in range(epochs_finetune):
        dbn_classifier.train()
        for data,target in train_loader:optimizer.zero_grad();output=dbn_classifier(data.to(DEVICE));loss=criterion(output,target.to(DEVICE));loss.backward();optimizer.step()
        if(epoch+1)%5==0:
            dbn_classifier.eval();correct,total=0,0
            with torch.no_grad():
                for data,target in test_loader:output=dbn_classifier(data.to(DEVICE));predicted=(torch.sigmoid(output)>0.5).float();total+=target.size(0);correct+=(predicted==target.to(DEVICE)).sum().item()
            print(f"  Fine-tune Epoch {epoch+1}/{epochs_finetune}, Test Accuracy: {100*correct/total:.2f}%")
    print("\nDBN fine-tuning finished.")

# ============================================================================
# 3. Main Execution Block
# ============================================================================

if __name__ == '__main__':
    # Script 1: Train a standalone RBM with specific hyperparameters
    train_rbm_with_gibbs(
        epochs=80,
        learning_rate=1,
        hidden_dim=32,
        k_gibbs_steps=15
    )

    # Script 2: Train a DBN with specific hyperparameters
    img_height, img_width = 5, 5
    data_cpu = generate_bars_and_stripes(img_height, img_width, 2000)
    labels_cpu = create_labels_for_bars_and_stripes(data_cpu, img_height, img_width)
    x_train, x_test, y_train, y_test = train_test_split(data_cpu, labels_cpu, test_size=0.2, random_state=42)
    train_dbn(
        x_train, y_train, x_test, y_test,
        hidden_sizes=[128, 64],
        epochs_pretrain=40,
        epochs_finetune=10,
        lr_pretrain=1,
        solver_iterations=30
    )
