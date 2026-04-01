import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from tqdm import tqdm
import torch
from torch.utils.data import random_split

# Import the unmodified classifiers (Expects reml.py to be in the same folder)
from reml import NadarayaWatsonClassifier, LocalizedNWC

# ==========================================
# HYPERPARAMETERS & CONFIGURATION
# ==========================================
GRAYSCALE = 1           # 0 for RGB (EuroSAT is naturally RGB)
LIPSCHITZ_L = 0.05      # Lipschitz constant
BANDWIDTH_H = 9.       # Bandwidth (Lamda)
LOCAL_K = 30            # Number of neighbors for Localized NWC
SUBSET_SIZE = None      # Subset for training (EuroSAT is 27k images, NWC is slow on full set)
TEST_SUBSET = 500       # Subset for testing

# EuroSAT Class Names
EUROSAT_CLASSES = {
    0: 'AnnualCrop', 1: 'Forest', 2: 'HerbaceousVeg', 3: 'Highway', 4: 'Industrial',
    5: 'Pasture', 6: 'PermanentCrop', 7: 'Residential', 8: 'River', 9: 'SeaLake'
}

# KERNEL DEFINITIONS
def K_epanechnikov(x, s: float = 1.):
    x = np.asarray(x)
    epanechnikov = np.where(x > 1., 1e-8, s * (1 - np.power(x, 2)))
    return epanechnikov

# ==========================================
# 1. DATA LOADING AND PREPROCESSING
# ==========================================
def get_eurosat_data():
    print(f"Loading EuroSAT Dataset... (Grayscale={bool(GRAYSCALE)})")
    
    # 1. Transform: Resize to 32x32 (EuroSAT is 64x64), optional Grayscale, convert to Tensor
    t_list = [transforms.Resize((32, 32))]
    if GRAYSCALE:
        t_list.append(transforms.Grayscale(num_output_channels=1))
    t_list.append(transforms.ToTensor())
    transform = transforms.Compose(t_list)

    # 2. Download Data
    # EuroSAT does not have built-in splits in torchvision, so we download the whole thing
    try:
        full_dataset = torchvision.datasets.EuroSAT(root='./data', download=True, transform=transform)
    except RuntimeError as e:
        print("Error downloading EuroSAT. You may need to install scipy: `pip install scipy`")
        raise e

    # 3. Create Train/Test Split (80/20)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_set, test_set = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    # 4. Convert to Numpy Arrays (Flattened)
    def dataset_to_numpy(dataset, limit=None):
        # Use a loader to handle batching and transforms efficiently
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        data_list, label_list = [], []
        count = 0
        for imgs, labels in loader:
            # Flatten: (B, C, H, W) -> (B, Features)
            # e.g., 32x32x3 -> 3072 features
            flat = imgs.view(imgs.size(0), -1).numpy()
            data_list.append(flat)
            label_list.append(labels.numpy())
            count += len(labels)
            if limit and count >= limit:
                break
        
        if not data_list: return np.array([]), np.array([])
        return np.concatenate(data_list)[:limit], np.concatenate(label_list)[:limit]

    print(f"Processing Training Data (Limit: {SUBSET_SIZE})...")
    X_train, y_train_raw = dataset_to_numpy(train_set, limit=SUBSET_SIZE)
    
    print(f"Processing Test Data (Limit: {TEST_SUBSET})...")
    X_test, y_test_raw = dataset_to_numpy(test_set, limit=TEST_SUBSET)

    # 5. Scaling (0 to 1) - Critical for NWC distance metrics
    print("Scaling Data...")
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 6. One-Hot Encoding for Labels (Required for NWC probability calc)
    print("One-Hot Encoding Labels...")
    enc = OneHotEncoder(sparse_output=False, categories='auto')
    # Fit on all possible classes (0-9) to ensure shape safety
    all_classes = np.arange(10).reshape(-1, 1)
    enc.fit(all_classes)
    Y_train = enc.transform(y_train_raw.reshape(-1, 1))

    return X_train, Y_train, y_train_raw, X_test, y_test_raw

# ==========================================
# 2. VISUALIZATION HELPERS
# ==========================================
def visualize_predictions(X, y_true, y_pred, bounds, title, img_shape):
    """Plots random samples with their prediction, truth, and uncertainty bound."""
    plt.figure(figsize=(14, 6))
    indices = np.random.choice(len(X), 10, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        
        # Reshape flattened vector back to image
        img = X[idx].reshape(img_shape)
        
        # Matplotlib expects (H, W, C) for RGB, but Torch gives (C, H, W)
        if img_shape[0] == 1: # Grayscale
            plt.imshow(img.transpose(1, 2, 0), cmap='gray')
        else: # RGB
            plt.imshow(img.transpose(1, 2, 0))
            
        plt.axis('off')
        
        # Color code: Green if correct, Red if wrong
        color = 'green' if y_pred[idx] == y_true[idx] else 'red'
        
        true_name = EUROSAT_CLASSES.get(y_true[idx], str(y_true[idx]))
        pred_name = EUROSAT_CLASSES.get(y_pred[idx], str(y_pred[idx]))
        
        info_text = f"True: {true_name}\nPred: {pred_name}\nBound: {bounds[idx]:.2f}"
        plt.title(info_text, color=color, fontsize=9, fontweight='bold')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    

def show_metrics(y_true, y_pred, classifier_name):
    print(f"\n--- {classifier_name} Metrics ---")
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    # Get class names present in the classification report
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    target_names = [EUROSAT_CLASSES[i] for i in unique_labels]
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0, target_names=target_names))
    
    # Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(include_values=False, cmap='viridis', ax=ax, xticks_rotation='vertical')
    plt.title(f"{classifier_name} Confusion Matrix")
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. MAIN EXPERIMENT
# ==========================================
def main():
    # Load Data
    X_train, Y_train, y_train_raw, X_test, y_test_raw = get_eurosat_data()
    
    channels = 1 if GRAYSCALE else 3
    img_shape = (channels, 32, 32)
    
    print(f"\nData Loaded. Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Parameters -> L: {LIPSCHITZ_L}, Lambda: {BANDWIDTH_H}")

    # -----------------------------------
    # REGULAR NADARAYA-WATSON
    # -----------------------------------
    print("\n[1/2] Running Regular Nadaraya-Watson Classifier...")
    print("Note: This is a lazy learner. No fitting. Computing predictions directly...")
    
    # Instantiate
    reg_nwc = NadarayaWatsonClassifier(
        bandwidth=BANDWIDTH_H, 
        lipschitz_constant=LIPSCHITZ_L, 
        ck=1., 
        kernel=K_epanechnikov
    )
    
    # Predict loop
    reg_preds = []
    reg_bounds = []
    
    # Iterate manually to use the predict(x, X, Y) signature from reml.py
    for i in tqdm(range(len(X_test)), desc="Regular NWC"):
        prob, bound = reg_nwc.predict(X_test[i], X_train, Y_train)
        pred_class = np.argmax(prob)
        reg_preds.append(pred_class)
        reg_bounds.append(bound)
        
    reg_preds = np.array(reg_preds)
    reg_bounds = np.array(reg_bounds)
    
    show_metrics(y_test_raw, reg_preds, "Regular NWC")
    visualize_predictions(X_test, y_test_raw, reg_preds, reg_bounds, "Regular NWC Sample Predictions", img_shape)

    # -----------------------------------
    # LOCALIZED NADARAYA-WATSON
    # -----------------------------------
    print("\n[2/2] Running Localized Nadaraya-Watson Classifier...")
    
    # Instantiate
    loc_nwc = LocalizedNWC(
        bandwidth=BANDWIDTH_H, 
        lipschitz_constant=LIPSCHITZ_L, 
        kernel=K_epanechnikov
    )
    
    # Fit (Localized NWC builds a Tree)
    print("Fitting Tree...")
    loc_nwc.fit(X_train, Y_train)
    
    # Predict loop
    print(f"Predicting with k={LOCAL_K}...")
    loc_preds = []
    loc_bounds = []
    
    for i in tqdm(range(len(X_test)), desc="Localized NWC"):
        prob, bound = loc_nwc.predict(X_test[i], k=LOCAL_K)
        pred_class = np.argmax(prob)
        loc_preds.append(pred_class)
        loc_bounds.append(bound)

    loc_preds = np.array(loc_preds)
    loc_bounds = np.array(loc_bounds)

    show_metrics(y_test_raw, loc_preds, "Localized NWC")
    visualize_predictions(X_test, y_test_raw, loc_preds, loc_bounds, "Localized NWC Sample Predictions", img_shape)

if __name__ == "__main__":
    main()