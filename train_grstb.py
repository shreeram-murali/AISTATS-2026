import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from tqdm import tqdm

# Specific imports for the requested preprocessing
from skimage import color, exposure, transform as sk_transform

# Import the unmodified classifiers
from reml import NadarayaWatsonClassifier, LocalizedNWC

# ==========================================
# HYPERPARAMETERS & CONFIGURATION
# ==========================================
LIPSCHITZ_L = 0.03      
BANDWIDTH_H = 7.5       
LOCAL_K = 20            
SUBSET_SIZE = None      # Limit training size for speed
TEST_SUBSET = 500       # Limit test size

# ==========================================
# 1. EXACT REFERENCE PREPROCESSING
# ==========================================
class CustomGTSRBPreprocess:
    """
    Implements the exact preprocessing pipeline from the reference snippet:
    1. HSV Histogram Equalization (V channel)
    2. Central Square Crop
    3. Resize to 32x32
    4. Manual Grayscale Conversion using specific weights
    5. Flattening
    """
    def __call__(self, img):
        # 1. Convert PIL (from torchvision) to Numpy RGB
        img = np.array(img)

        # 2. Histogram normalization in v channel
        hsv = color.rgb2hsv(img)
        hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
        img = color.hsv2rgb(hsv)

        # 3. Central square crop
        min_side = min(img.shape[:-1])
        centre = img.shape[0] // 2, img.shape[1] // 2
        img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
                  centre[1] - min_side // 2:centre[1] + min_side // 2,
                  :]

        # 4. Rescale to standard size
        img = sk_transform.resize(img, (32, 32))

        # 5. Grayscale Dot Product (Specific weights from reference)
        rgb_weights = [0.2989, 0.5870, 0.1140]
        # img is (32, 32, 3) at this point
        img = np.dot(img[...,:3], rgb_weights)

        # 6. Flatten
        # img becomes (32, 32) -> Flatten to (1024,)
        return img.flatten().astype(np.float32)

# ==========================================
# 2. DATA LOADING
# ==========================================
def K_epanechnikov(x, s: float = 1.):
    x = np.asarray(x)
    epanechnikov = np.where(x > 1., 1e-8, s * (1 - np.power(x, 2)))
    return epanechnikov

def get_processed_data():
    print("Loading GTSRB with Custom Scikit-Image Preprocessing...")
    
    # Use the custom class instead of standard transforms
    custom_transform = CustomGTSRBPreprocess()

    # Download/Load Data
    train_set = torchvision.datasets.GTSRB(root='./data', split='train', download=True, transform=custom_transform)
    test_set = torchvision.datasets.GTSRB(root='./data', split='test', download=True, transform=custom_transform)

    # Helper to pull data out of the dataset object into Numpy arrays
    def extract_numpy(dataset, limit=None):
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        data_list, label_list = [], []
        count = 0
        
        print(f"Extracting and processing {limit if limit else 'all'} images...")
        for imgs, labels in loader:
            # imgs are already flattened numpy arrays (float32) due to our custom_transform
            data_list.append(imgs.numpy())
            label_list.append(labels.numpy())
            
            count += len(labels)
            if limit and count >= limit:
                break
                
        X = np.concatenate(data_list)[:limit]
        y_raw = np.concatenate(label_list)[:limit]
        return X, y_raw

    X_train, y_train_raw = extract_numpy(train_set, limit=SUBSET_SIZE)
    X_test, y_test_raw = extract_numpy(test_set, limit=TEST_SUBSET)

    # One-Hot Encoding (Manual, matching the reference style)
    # The reference used: Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]
    NUM_CLASSES = 43
    Y_train = np.eye(NUM_CLASSES)[y_train_raw]

    return X_train, Y_train, y_train_raw, X_test, y_test_raw

# ==========================================
# 3. VISUALIZATION & METRICS
# ==========================================
def visualize_predictions(X, y_true, y_pred, bounds, title):
    plt.figure(figsize=(14, 6))
    indices = np.random.choice(len(X), 10, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        
        # Reshape flattened vector back to image (32x32 grayscale)
        img = X[idx].reshape(32, 32)
        plt.imshow(img, cmap='gray')
            
        plt.axis('off')
        color = 'green' if y_pred[idx] == y_true[idx] else 'red'
        info_text = f"True: {y_true[idx]}\nPred: {y_pred[idx]}\nBound: {bounds[idx]:.2f}"
        plt.title(info_text, color=color, fontsize=10, fontweight='bold')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def show_metrics(y_true, y_pred, classifier_name):
    print(f"\n--- {classifier_name} Metrics ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    
    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(include_values=False, cmap='viridis', ax=ax)
    plt.title(f"{classifier_name} Confusion Matrix")
    plt.show()

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    # 1. Get Data
    X_train, Y_train, y_train_raw, X_test, y_test_raw = get_processed_data()
    
    print(f"\nData Shape Check:")
    print(f"X_train: {X_train.shape} (Values: {X_train.min():.2f} to {X_train.max():.2f})")
    print(f"Y_train: {Y_train.shape}")

    # 2. Regular NWC
    print("\n[1/2] Running Regular Nadaraya-Watson Classifier...")
    reg_nwc = NadarayaWatsonClassifier(
        bandwidth=BANDWIDTH_H, 
        lipschitz_constant=LIPSCHITZ_L, 
        ck=1., 
        kernel=K_epanechnikov
    )
    
    reg_preds, reg_bounds = [], []
    for i in tqdm(range(len(X_test)), desc="Regular NWC"):
        prob, bound = reg_nwc.predict(X_test[i], X_train, Y_train)
        reg_preds.append(np.argmax(prob))
        reg_bounds.append(bound)
        
    show_metrics(y_test_raw, np.array(reg_preds), "Regular NWC")
    visualize_predictions(X_test, y_test_raw, np.array(reg_preds), np.array(reg_bounds), "Regular NWC (Custom Preprocess)")

    # 3. Localized NWC
    print("\n[2/2] Running Localized Nadaraya-Watson Classifier...")
    loc_nwc = LocalizedNWC(
        bandwidth=BANDWIDTH_H, 
        lipschitz_constant=LIPSCHITZ_L, 
        kernel=K_epanechnikov
    )
    
    loc_nwc.fit(X_train, Y_train)
    
    loc_preds, loc_bounds = [], []
    for i in tqdm(range(len(X_test)), desc="Localized NWC"):
        prob, bound = loc_nwc.predict(X_test[i], k=LOCAL_K)
        loc_preds.append(np.argmax(prob))
        loc_bounds.append(bound)

    show_metrics(y_test_raw, np.array(loc_preds), "Localized NWC")
    visualize_predictions(X_test, y_test_raw, np.array(loc_preds), np.array(loc_bounds), "Localized NWC (Custom Preprocess)")

if __name__ == "__main__":
    main()