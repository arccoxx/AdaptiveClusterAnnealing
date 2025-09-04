import numpy as np
import time
import math
import os

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Attempt to import cupy and provide instructions if it fails.
try:
    import cupy as cp
    print("✅ CuPy (GPU) backend found.")
except ImportError:
    print("⚠️ CuPy not found. This script requires a CUDA-enabled GPU and CuPy.")
    print("Install with: pip install cupy-cudaXXX (e.g., cupy-cuda12x)")
    exit()

import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ============================================================================
# 1. Stable Gibbs Sampler using pure CuPy
# ============================================================================

def cupy_sigmoid(x):
    """Numerically stable sigmoid function for CuPy arrays."""
    return 1. / (1. + cp.exp(-x))

def cupy_gibbs_sampling(v0, W_gpu, b_gpu, c_gpu, k=10):
    """
    Performs k steps of block Gibbs sampling, starting from the input data.
    """
    vk = v0
    for _ in range(k):
        phk = cupy_sigmoid(cp.dot(vk, W_gpu) + c_gpu)
        hk = (cp.random.rand(*phk.shape) < phk).astype(cp.float32)
        pvk = cupy_sigmoid(cp.dot(hk, W_gpu.T) + b_gpu)
        vk = (cp.random.rand(*pvk.shape) < pvk).astype(cp.float32)
    
    phk_final = cupy_sigmoid(cp.dot(vk, W_gpu) + c_gpu)
    return vk, phk_final

# ============================================================================
# 2. RBM Training Script
# ============================================================================

def generate_bars_and_stripes(height, width, num_samples):
    """Generates the 'bars and stripes' synthetic dataset."""
    data = np.zeros((num_samples, height * width), dtype=np.float32)
    for i in range(num_samples):
        if np.random.rand() > 0.5: # Horizontal
            img = np.zeros((height, width), dtype=np.float32); row_indices = np.random.choice(height, np.random.randint(1, height + 1), replace=False); img[row_indices, :] = 1; data[i, :] = img.flatten()
        else: # Vertical
            img = np.zeros((height, width), dtype=np.float32); col_indices = np.random.choice(width, np.random.randint(1, width + 1), replace=False); img[:, col_indices] = 1; data[i, :] = img.flatten()
    return data

def train_rbm_with_gibbs_sampling():
    print("\n" + "="*50)
    print("--- Training Standalone RBM with CuPy Gibbs Sampling ---")
    print("="*50)
    
    V_dim, H_dim = 16, 16
    epochs = 100
    learning_rate = 0.1
    batch_size = 20
    k_gibbs_steps = 10
    num_data_points = 1000
    
    data = generate_bars_and_stripes(4, 4, num_data_points)
    
    W_gpu = cp.asarray(np.random.randn(V_dim, H_dim).astype(np.float32) * 0.1)
    b_gpu = cp.asarray(np.zeros(V_dim, dtype=np.float32))
    c_gpu = cp.asarray(np.zeros(H_dim, dtype=np.float32))

    print(f"Starting RBM training for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_error = 0.0
        for i in range(0, len(data), batch_size):
            batch_gpu = cp.asarray(data[i:i+batch_size])
            
            v_pos = batch_gpu
            h_pos_prob = cupy_sigmoid(cp.dot(v_pos, W_gpu) + c_gpu)
            v_neg, h_neg_prob = cupy_gibbs_sampling(v_pos, W_gpu, b_gpu, c_gpu, k=k_gibbs_steps)
            
            dW = cp.dot(v_pos.T, h_pos_prob) - cp.dot(v_neg.T, h_neg_prob)
            db = cp.sum(v_pos - v_neg, axis=0)
            dc = cp.sum(h_pos_prob - h_neg_prob, axis=0)
            
            W_gpu += (learning_rate / batch_size) * dW
            b_gpu += (learning_rate / batch_size) * db
            c_gpu += (learning_rate / batch_size) * dc
            
            epoch_error += cp.sum((v_pos - v_neg)**2).get()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Reconstruction Error: {epoch_error / len(data):.4f}")

    print("Standalone RBM training finished.")
    
    W_final = W_gpu.get()
    plt.figure(figsize=(8, 8))
    plt.suptitle("Standalone RBM Learned Features (Weights)", fontsize=16)
    for i in range(H_dim):
        plt.subplot(4, 4, i + 1)
        plt.imshow(W_final[:, i].reshape(4, 4), cmap='gray', interpolation='nearest')
        plt.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# ============================================================================
# 3. DBN Training Script with Fine-Tuning
# ============================================================================

def create_labels_for_bars_and_stripes(data, height, width):
    """Creates binary labels: 0 for mostly horizontal, 1 for mostly vertical."""
    labels = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        img = data[i].reshape(height, width)
        if np.sum(np.var(img, axis=1)) > np.sum(np.var(img, axis=0)):
            labels[i] = 0
        else:
            labels[i] = 1
    return labels

def train_dbn_with_finetuning(x_train_cpu, y_train_cpu, x_test_cpu, y_test_cpu):
    print("\n" + "="*50)
    print("--- Training DBN with Pre-training and Fine-Tuning ---")
    print("="*50)

    hidden_sizes = [256, 128] 
    epochs_pretrain = 50
    epochs_finetune = 30
    learning_rate = 0.05
    batch_size = 32
    k_gibbs_steps = 3

    print("\n--- PHASE 1: Greedy Layer-wise Pre-training ---")
    current_input_gpu = cp.asarray(x_train_cpu)
    trained_layers = []
    
    for i, h_size in enumerate(hidden_sizes):
        V_dim = current_input_gpu.shape[1]
        print(f"\n--- Pre-training DBN Layer {i+1} (RBM: {V_dim} -> {h_size}) ---")
        
        W_gpu = cp.asarray(np.random.randn(V_dim, h_size).astype(np.float32) * 0.1)
        b_gpu = cp.asarray(np.zeros(V_dim, dtype=np.float32))
        c_gpu = cp.asarray(np.zeros(h_size, dtype=np.float32))
        
        for epoch in range(epochs_pretrain):
            for j in range(0, current_input_gpu.shape[0], batch_size):
                batch = current_input_gpu[j:j+batch_size]
                v_pos, h_pos_prob = batch, cupy_sigmoid(cp.dot(batch, W_gpu) + c_gpu)
                v_neg, h_neg_prob = cupy_gibbs_sampling(v_pos, W_gpu, b_gpu, c_gpu, k=k_gibbs_steps)
                dW, db, dc = cp.dot(v_pos.T, h_pos_prob) - cp.dot(v_neg.T, h_neg_prob), cp.sum(v_pos - v_neg, axis=0), cp.sum(h_pos_prob - h_neg_prob, axis=0)
                W_gpu += (learning_rate / batch_size) * dW; b_gpu += (learning_rate / batch_size) * db; c_gpu += (learning_rate / batch_size) * dc
            
            if (epoch + 1) % 25 == 0:
                 print(f"Layer {i+1}, Pre-train Epoch {epoch+1}/{epochs_pretrain}")
        
        trained_layers.append({'W': W_gpu.get(), 'c': c_gpu.get()})
        current_input_gpu = cupy_sigmoid(cp.dot(current_input_gpu, W_gpu) + c_gpu)

    print("\nDBN pre-training finished.")

    print("\n--- PHASE 2: Supervised Fine-tuning with Backpropagation ---")
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(x_train_cpu.shape[1],))])
    
    for i, layer_params in enumerate(trained_layers):
        W, c = layer_params['W'], layer_params['c']
        
        # --- THIS IS THE FIX ---
        # 1. Create the layer first.
        layer = tf.keras.layers.Dense(W.shape[1], activation='sigmoid')
        model.add(layer)
        # 2. Then, set the weights.
        layer.set_weights([W, c])
        
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    print(f"\nStarting fine-tuning for {epochs_finetune} epochs...")
    model.fit(x_train_cpu, y_train_cpu, batch_size=batch_size, epochs=epochs_finetune, validation_data=(x_test_cpu, y_test_cpu), verbose=1)

    print("\nDBN fine-tuning finished.")
    final_loss, final_acc = model.evaluate(x_test_cpu, y_test_cpu, verbose=0)
    print(f"\nFinal Test Accuracy of the DBN: {final_acc*100:.2f}%")

# ============================================================================
# 4. Standard Neural Network Training on MNIST
# ============================================================================

def train_nn_on_mnist(samples_per_digit=None):
    """
    Trains a standard Feed-Forward Neural Network on the MNIST dataset.
    """
    print("\n" + "="*50)
    print("--- Training a Standard Neural Network on MNIST ---")
    print("="*50)
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
    
    if samples_per_digit is not None:
        print(f"Downsampling training data to {samples_per_digit} sample(s) per digit.")
        x_tiny_train, y_tiny_train = [], []
        for digit in range(10):
            digit_indices = np.where(y_train == digit)[0]
            num_samples = min(samples_per_digit, len(digit_indices))
            selected_indices = np.random.choice(digit_indices, num_samples, replace=False)
            x_tiny_train.append(x_train[selected_indices])
            y_tiny_train.append(y_train[selected_indices])
        
        x_train = np.concatenate(x_tiny_train, axis=0)
        y_train = np.concatenate(y_tiny_train, axis=0)
        
        p = np.random.permutation(len(x_train))
        x_train, y_train = x_train[p], y_train[p]
        
        print(f"Total size of the new training set: {len(x_train)} images.")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("Starting MNIST model training...")
    if samples_per_digit is not None:
        model.fit(x_train, y_train, batch_size=1, epochs=50, validation_split=0.0, verbose=0)
    else:
        model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1, verbose=0)
        
    print("MNIST model training finished.")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")

# ============================================================================
# 5. Main Execution Block
# ============================================================================

if __name__ == '__main__':
    
    # --- Script 1: Train a standalone RBM ---
    train_rbm_with_gibbs_sampling()
    
    # --- Script 2: Train and Fine-tune a DBN ---
    img_height, img_width = 5, 5
    data_cpu = generate_bars_and_stripes(img_height, img_width, num_samples=2000)
    labels_cpu = create_labels_for_bars_and_stripes(data_cpu, img_height, img_width)
    x_train, x_test, y_train, y_test = train_test_split(
        data_cpu, labels_cpu, test_size=0.2, random_state=42
    )
    train_dbn_with_finetuning(x_train, y_train, x_test, y_test)
    
    # --- Script 3: Train a standard Neural Network on MNIST with 1 sample per digit ---
    train_nn_on_mnist(samples_per_digit=100)
