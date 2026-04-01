import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from tqdm import tqdm

try:
    import matplotlib_inline
    matplotlib_inline.backend_inline.set_matplotlib_formats('retina')
except ImportError:
    matplotlib_inline = None

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update(
    {
        'text.usetex': True,
        'font.family': 'sans-serif',
    }
)

# Specific imports for the requested preprocessing
from skimage import color, exposure, transform as sk_transform

# Import the unmodified classifiers
from reml import NadarayaWatsonClassifier, LocalizedNWC

# ==========================================
# AISTATS PAPER FIGURE SETTINGS
# ==========================================
page_width_pt = 487.8225
column_width_pt = 234.8775


def set_figure_size(width_pt, fraction=0.9, subplots=(1, 1), ratio=(5**0.5 - 1) / 2):
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


base_figure_size = set_figure_size(page_width_pt, 0.9, (1, 1))

# ==========================================
# HYPERPARAMETERS & CONFIGURATION
# ==========================================
LIPSCHITZ_L = 0.02      # Adjusted for binary separation
BANDWIDTH_H = 7.5     # Slightly tighter bandwidth
LOCAL_K = 20            
SUBSET_SIZE = None      # None = Use all available binary data (it's much smaller now)
TEST_SUBSET = None      # None = Use all available binary test data

# Mapping for our new binary problem
# 0: Class 6 (End of 80km/h limit -> "Go")
# 1: Class 14 (Stop -> "Stop")
BINARY_CLASS_NAMES = ['Go', 'Stop']

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
# 2. DATA LOADING & BINARY FILTERING
# ==========================================
def K_epanechnikov(x, s: float = 1.):
    x = np.asarray(x)
    epanechnikov = np.where(x > 1., 1e-8, s * (1 - np.power(x, 2)))
    return epanechnikov

def get_binary_data():
    print("Loading GTSRB and Filtering for Binary Classification (Class 6 vs 14)...")
    
    custom_transform = CustomGTSRBPreprocess()

    # Download/Load Data
    train_set = torchvision.datasets.GTSRB(root='./data', split='train', download=True, transform=custom_transform)
    test_set = torchvision.datasets.GTSRB(root='./data', split='test', download=True, transform=custom_transform)

    def extract_and_filter(dataset, limit=None):
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        data_list, label_list = [], []
        count = 0
        
        # 1. Extract all data
        for imgs, labels in loader:
            data_list.append(imgs.numpy())
            label_list.append(labels.numpy())
        
        X = np.concatenate(data_list)
        y = np.concatenate(label_list)

        # 2. Filter for only Class 6 and Class 14
        # Class 6: End of speed limit (Go)
        # Class 14: Stop
        mask = (y == 6) | (y == 14)
        X_filtered = X[mask]
        y_filtered = y[mask]

        # 3. Remap labels to 0 and 1
        # 6 -> 0
        # 14 -> 1
        y_binary = np.where(y_filtered == 6, 0, 1)

        if limit:
            return X_filtered[:limit], y_binary[:limit]
        return X_filtered, y_binary

    X_train, y_train_raw = extract_and_filter(train_set, limit=SUBSET_SIZE)
    X_test, y_test_raw = extract_and_filter(test_set, limit=TEST_SUBSET)

    # One-Hot Encoding for Binary (2 classes)
    # [1, 0] = Go, [0, 1] = Stop
    NUM_CLASSES = 2
    Y_train = np.eye(NUM_CLASSES)[y_train_raw]

    return X_train, Y_train, y_train_raw, X_test, y_test_raw

# ==========================================
# 3. VISUALIZATION & METRICS
# ==========================================
def visualize_predictions(X, y_true, y_pred, bounds, title):
    plt.figure(figsize=(14, 6))
    indices = np.random.choice(len(X), min(10, len(X)), replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        
        # Reshape flattened vector back to image (32x32 grayscale)
        img = X[idx].reshape(32, 32)
        plt.imshow(img, cmap='gray')
            
        plt.axis('off')
        color = 'green' if y_pred[idx] == y_true[idx] else 'red'
        
        true_name = BINARY_CLASS_NAMES[y_true[idx]]
        pred_name = BINARY_CLASS_NAMES[y_pred[idx]]
        
        info_text = f"True: {true_name}\nPred: {pred_name}\nBound: {bounds[idx]:.2f}"
        plt.title(info_text, color=color, fontsize=10, fontweight='bold')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def show_metrics(y_true, y_pred, classifier_name, ax=None, show_figure=True):
    print(f"\n--- {classifier_name} Metrics ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    
    # Classification Report
    print(classification_report(y_true, y_pred, target_names=BINARY_CLASS_NAMES, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    if ax is not None:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(include_values=True, cmap='Blues', ax=ax, values_format='d', colorbar=False)
        ax.set_title(f"{classifier_name} Confusion Matrix")
    elif show_figure:
        fig, ax_fig = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(include_values=True, cmap='Blues', ax=ax_fig, values_format='d')
        plt.title(f"{classifier_name} Confusion Matrix")
        plt.show()

    return cm


def create_paper_figure(X, y_true, reg_preds, loc_preds, reg_cm, loc_cm, filename="gtsrb_binary_paper_figure.pdf"):
    # Full-width, 1x4 layout
    fig_size = set_figure_size(page_width_pt, fraction=0.9, subplots=(1, 4))
    fig, axes = plt.subplots(1, 4, figsize=(8, 4))

    # Select representative Go and Stop examples
    rng = np.random.default_rng(0)

    go_indices = np.where(y_true == 0)[0]
    stop_indices = np.where(y_true == 1)[0]

    if len(go_indices) == 0 or len(stop_indices) == 0:
        raise ValueError("Cannot create figure: one of the binary classes has zero samples in y_true.")

    go_idx = rng.choice(go_indices)
    stop_idx = rng.choice(stop_indices)

    # 1. Go sign
    axes[0].imshow(X[go_idx].reshape(32, 32), cmap='gray')
    axes[0].set_title(BINARY_CLASS_NAMES[0])
    axes[0].axis('off')

    # 2. Stop sign
    axes[1].imshow(X[stop_idx].reshape(32, 32), cmap='gray')
    axes[1].set_title(BINARY_CLASS_NAMES[1])
    axes[1].axis('off')

    # 3. Confusion matrix for Regular NWC
    disp_reg = ConfusionMatrixDisplay(confusion_matrix=reg_cm, display_labels=[0, 1])
    disp_reg.plot(include_values=True, cmap='Blues', ax=axes[2], values_format='d', colorbar=False)
    axes[2].set_title("Regular NWC")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("")
    # axes[2].set_xticks([])
    # axes[2].set_yticks([])

    # 4. Confusion matrix for Localized NWC
    disp_loc = ConfusionMatrixDisplay(confusion_matrix=loc_cm, display_labels=[0, 1])
    disp_loc.plot(include_values=True, cmap='Blues', ax=axes[3], values_format='d', colorbar=False)
    axes[3].set_title("Localized NWC")
    axes[3].set_xlabel("")
    axes[3].set_ylabel("")
    # axes[3].set_xticks([])
    # axes[3].set_yticks([])

    fig.tight_layout()

    output_dir = os.path.dirname(__file__)
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"Saved paper figure to: {output_path}")

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    # 1. Get Data
    X_train, Y_train, y_train_raw, X_test, y_test_raw = get_binary_data()
    
    print(f"\nBinary Data Shape Check:")
    print(f"X_train: {X_train.shape}")
    print(f"Y_train: {Y_train.shape}")
    print(f"Class Distribution Train: {np.bincount(y_train_raw)} (0=Go, 1=Stop)")

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
        
    reg_cm = show_metrics(y_test_raw, np.array(reg_preds), "Regular NWC", show_figure=False)
    visualize_predictions(X_test, y_test_raw, np.array(reg_preds), np.array(reg_bounds), "Regular NWC (Binary)")

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

    loc_cm = show_metrics(y_test_raw, np.array(loc_preds), "Localized NWC", show_figure=False)
    visualize_predictions(X_test, y_test_raw, np.array(loc_preds), np.array(loc_bounds), "Localized NWC (Binary)")

    # 4. Create full-width paper figure: Go, Stop, and both confusion matrices
    create_paper_figure(
        X_test,
        y_test_raw,
        np.array(reg_preds),
        np.array(loc_preds),
        reg_cm,
        loc_cm,
        filename="gtsrb_binary_paper_figure.pdf",
    )

if __name__ == "__main__":
    main()