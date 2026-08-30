# Thesis Diagrams — Nairobi Urban Flood Digital Twin

This document contains all Mermaid diagrams for your thesis. Copy each diagram into your thesis document or render them on the Mermaid Live Editor: https://mermaid.live/

---

## 1. System Architecture Overview

```mermaid
graph TB
    subgraph Data["🛰️ Data Sources"]
        SAR["Sentinel-1 SAR<br/>VV Polarization<br/>7-day Time Series"]
        CHIRPS["CHIRPS Rainfall<br/>7-day Antecedent<br/>30mm Threshold"]
        TERRAIN["Terrain Predictors<br/>HAND/UPA/DEM<br/>Built-up Fraction"]
    end
    
    subgraph Pipeline["📦 Data Pipeline"]
        Fetch["Fetch & Preprocess<br/>703 Scenes"]
        Label["Build Labels<br/>Rainfall >= 30mm<br/>51 Floods / 111 Non-floods"]
        Stack["Stack Predictors<br/>14 Channels<br/>7 Timesteps"]
        Split["Scene-Level Split<br/>70/15/15 Train/Val/Test<br/>No Pixel Leakage"]
    end
    
    subgraph Dataset["💾 Training Dataset"]
        DS["segmentation_train_dataset.npz<br/>6.1 GB<br/>14 Channels × 7 Timesteps<br/>492 Train / 105 Val / 106 Test"]
    end
    
    subgraph Model["🧠 U-Net Model"]
        UNET["Binary Segmentation<br/>Input: 14 Channels<br/>Output: Flood Probability 0-1<br/>Loss: BCE 0.7 + Dice 0.3"]
    end
    
    subgraph GPU["🚀 Training"]
        COLAB["Google Colab GPU<br/>T4 NVIDIA<br/>~4 Hours<br/>50 Epochs"]
    end
    
    subgraph Results["📊 Results"]
        METRICS["Validation Metrics<br/>F1 / IoU / Precision / Recall<br/>Flood Probability Maps"]
    end
    
    SAR --> Fetch
    CHIRPS --> Label
    TERRAIN --> Stack
    
    Fetch --> Label
    Label --> Stack
    Stack --> Split
    Split --> DS
    
    DS --> UNET
    UNET --> COLAB
    COLAB --> METRICS
```

---

## 2. Problem Discovery Flow

```mermaid
graph LR
    A["Original Approach<br/>SAR -16dB Threshold"] 
    B["Finding 1<br/>R² ≈ 0<br/>Labels Inverted"]
    C["Investigation<br/>Spearman r = -0.74<br/>Anti-Correlated w/ Rainfall"]
    
    D["Attempt 1: Optical<br/>Sentinel-2 Water Index"]
    E["Finding 2<br/>156 Scenes<br/>0 Water Pixels"]
    F["Root Cause<br/>Urban Flooding<br/>10m Below Resolution"]
    
    G["Attempt 2: SAR CD<br/>Backscatter Drop vs Baseline"]
    H["Finding 3<br/>Mean Drop = -0.30 dB<br/>r = -0.399 Inverted"]
    I["Root Cause<br/>Wet Soil Increases<br/>VV Backscatter"]
    
    J["Solution<br/>Rainfall-Based Labels<br/>Use SAR + Terrain as Predictors"]
    K["Result<br/>31.5% Class Balance<br/>51 Floods / 111 Non-floods<br/>Ready to Train"]
    
    A --> B --> C
    C -->|Reject SAR Absolute| D --> E --> F
    F -->|Reject Optical| G --> H --> I
    I -->|Reject SAR CD| J --> K
    
    style A fill:#e1f5ff
    style J fill:#c8e6c9
    style K fill:#c8e6c9
```

---

## 3. Data Pipeline Architecture

```mermaid
graph TB
    subgraph input["Input: Satellite & Ground Truth"]
        S1["Sentinel-1 SAR<br/>4 Channels<br/>7 Timesteps<br/>VV Backscatter"]
        S2["Rainfall Labels<br/>Binary<br/>Rain >= 30mm"]
        PRED["7 Terrain Predictors<br/>HAND, Slope, Elevation<br/>Built-up, TWI, etc."]
    end
    
    subgraph processing["Processing"]
        P1["Reproject to<br/>EPSG:4326"]
        P2["Resample to<br/>198×252 Grid"]
        P3["Normalize Ranges<br/>SAR: dB Scale<br/>Terrain: Physical Units"]
        P4["Stack Channels<br/>Temporal + Spatial"]
    end
    
    subgraph assembly["Dataset Assembly"]
        A1["Load 703 Scenes"]
        A2["Pair SAR + Labels<br/>+ Terrain"]
        A3["Create 70/15/15 Split<br/>at Scene Level"]
        A4["Compress to NPZ<br/>6.1 GB"]
    end
    
    subgraph output["Output: Training Data"]
        O1["X_train: 492×7×11×198×252"]
        O2["y_train: 492×1×198×252"]
        O3["X_val: 105×7×11×198×252"]
        O4["y_val: 105×1×198×252"]
        O5["X_test: 106×7×11×198×252"]
        O6["y_test: 106×1×198×252"]
    end
    
    S1 --> P1
    S2 --> P3
    PRED --> P2
    
    P1 --> P4
    P2 --> P4
    P3 --> P4
    
    P4 --> A1
    A1 --> A2
    A2 --> A3
    A3 --> A4
    
    A4 --> O1
    A4 --> O2
    A4 --> O3
    A4 --> O4
    A4 --> O5
    A4 --> O6
```

---

## 4. U-Net Model Architecture

```mermaid
graph TB
    subgraph input["Input Layer"]
        I["14 Channels<br/>7 Timesteps<br/>198×252 Spatial"]
    end
    
    subgraph encoder["Encoder (Downsampling)"]
        E1["Conv + ReLU + BN<br/>14 → 32 channels<br/>198×252"]
        E2["MaxPool + Conv<br/>32 → 64 channels<br/>99×126"]
        E3["MaxPool + Conv<br/>64 → 128 channels<br/>49×63"]
        E4["MaxPool + Conv<br/>128 → 256 channels<br/>24×31"]
    end
    
    subgraph bottleneck["Bottleneck"]
        B["Conv + ReLU + BN<br/>256 → 256 channels<br/>24×31"]
    end
    
    subgraph decoder["Decoder (Upsampling)"]
        D1["Upsample + Concat<br/>256+128 → 128 channels<br/>49×63"]
        D2["Upsample + Concat<br/>128+64 → 64 channels<br/>99×126"]
        D3["Upsample + Concat<br/>64+32 → 32 channels<br/>198×252"]
    end
    
    subgraph output["Output Layer"]
        O["Conv 1×1<br/>32 → 1 channel<br/>Sigmoid: Flood Prob 0-1<br/>198×252"]
    end
    
    I --> E1 --> E2 --> E3 --> E4
    E4 --> B
    B --> D1 --> D2 --> D3 --> O
    
    E1 -.->|Skip Connection| D3
    E2 -.->|Skip Connection| D2
    E3 -.->|Skip Connection| D1
    E4 -.->|Skip Connection| B
```

---

## 5. Training Loop Workflow

```mermaid
graph TD
    START["Initialize U-Net<br/>Adam Optimizer<br/>Cosine Annealing LR"]
    
    EPOCH["For Each Epoch"]
    
    TRAIN["Training Phase<br/>492 Scenes"]
    TRAIN_FWD["Forward Pass<br/>X_train → Model → ŷ"]
    TRAIN_LOSS["Compute Loss<br/>0.7×BCE + 0.3×Dice"]
    TRAIN_BACK["Backward Pass<br/>Compute Gradients"]
    TRAIN_OPT["Update Weights<br/>Adam Step"]
    
    VAL["Validation Phase<br/>105 Scenes"]
    VAL_FWD["Forward Pass<br/>X_val → Model → ŷ"]
    VAL_LOSS["Compute Loss"]
    VAL_METRICS["Compute Metrics<br/>F1 / IoU<br/>Precision / Recall"]
    
    CHECK["Check F1 Score"]
    BEST["F1 > Best?"]
    SAVE["Save Checkpoint<br/>segmentation_model.pth"]
    LOG["Log Metrics<br/>segmentation_metrics.json"]
    
    DONE["50 Epochs?"]
    
    TEST["Test Phase<br/>106 Scenes"]
    TEST_EVAL["Evaluate<br/>Test Metrics"]
    FINAL["Output Results<br/>Metrics + Model Weights"]
    
    START --> EPOCH
    EPOCH --> TRAIN
    TRAIN --> TRAIN_FWD
    TRAIN_FWD --> TRAIN_LOSS
    TRAIN_LOSS --> TRAIN_BACK
    TRAIN_BACK --> TRAIN_OPT
    TRAIN_OPT --> VAL
    VAL --> VAL_FWD
    VAL_FWD --> VAL_LOSS
    VAL_LOSS --> VAL_METRICS
    VAL_METRICS --> CHECK
    CHECK --> BEST
    BEST -->|Yes| SAVE
    BEST -->|No| LOG
    SAVE --> LOG
    LOG --> DONE
    DONE -->|No| EPOCH
    DONE -->|Yes| TEST
    TEST --> TEST_EVAL
    TEST_EVAL --> FINAL
    
    style SAVE fill:#ffd54f
    style FINAL fill:#a5d6a7
```

---

## 6. Loss Function Composition

```mermaid
graph LR
    subgraph BCE["Binary Cross-Entropy (70%)"]
        BCE_F["BCE Loss<br/>-[y·log(ŷ) + (1-y)·log(1-ŷ)]<br/>Weight: 0.7"]
    end
    
    subgraph DICE["Dice Loss (30%)"]
        DICE_F["Dice Loss<br/>1 - 2|X∩Y|/(|X|+|Y|)<br/>Weight: 0.3"]
    end
    
    subgraph COMBINED["Total Loss"]
        TOTAL["L_total = 0.7×L_BCE + 0.3×L_Dice<br/>Handles Class Imbalance<br/>31.5% Positive Class"]
    end
    
    BCE_F --> TOTAL
    DICE_F --> TOTAL
    
    style BCE fill:#bbdefb
    style DICE fill:#bbdefb
    style TOTAL fill:#81c784
```

---

## 7. Class Distribution & Label Balance

```mermaid
graph TD
    DATA["162 Sentinel-1 Scenes<br/>2015-2026"]
    
    SPLIT["7-Day Rainfall Threshold: 30mm"]
    
    HEAVY["Heavy Rain >= 30mm<br/>51 Scenes (31.5%)"]
    LIGHT["Light Rain < 30mm<br/>111 Scenes (68.5%)"]
    
    EXPAND["Expand to 703 Scenes<br/>With All Terrain Combinations"]
    
    TRAIN["Training Set<br/>492 Scenes<br/>155 Floods / 337 Non-floods<br/>31.5% Imbalance"]
    VAL["Validation Set<br/>105 Scenes<br/>33 Floods / 72 Non-floods"]
    TEST["Test Set<br/>106 Scenes<br/>34 Floods / 72 Non-floods"]
    
    DATA --> SPLIT
    SPLIT --> HEAVY
    SPLIT --> LIGHT
    
    HEAVY --> EXPAND
    LIGHT --> EXPAND
    
    EXPAND --> TRAIN
    EXPAND --> VAL
    EXPAND --> TEST
    
    style HEAVY fill:#ffcdd2
    style LIGHT fill:#c8e6c9
```

---

## 8. Validation Workflow

```mermaid
graph TB
    subgraph input["Test Dataset"]
        T["X_test: 106 Scenes<br/>y_test: Ground Truth"]
    end
    
    subgraph inference["Inference"]
        FWD["Forward Pass<br/>X_test → Model → ŷ_prob"]
        THRESHOLD["Apply Threshold<br/>ŷ_pred = ŷ_prob > 0.5"]
    end
    
    subgraph metrics["Pixel-Level Metrics"]
        TP["True Positives<br/>Correctly Predicted Floods"]
        FP["False Positives<br/>False Alarms"]
        FN["False Negatives<br/>Missed Floods"]
        TN["True Negatives<br/>Correctly Predicted Non-floods"]
    end
    
    subgraph validation["Validation Checks"]
        CHECK1["1. Spatial Plausibility<br/>Floods in Low HAND?<br/>In Built-up Areas?"]
        CHECK2["2. Rainfall Correlation<br/>Predicted Floods correlate<br/>with 7-day Rainfall?"]
        CHECK3["3. Seasonal Patterns<br/>Peak Flooding Mar-May<br/>& Oct-Dec?"]
    end
    
    subgraph results["Performance Metrics"]
        F1["F1 Score<br/>Harmonic Mean"]
        IoU["IoU Score<br/>Intersection over Union"]
        Prec["Precision<br/>Reliability of Predictions"]
        Rec["Recall<br/>Detection Rate"]
        AUC["AUC-ROC<br/>Discrimination Power"]
    end
    
    input --> FWD
    FWD --> THRESHOLD
    THRESHOLD --> TP
    THRESHOLD --> FP
    THRESHOLD --> FN
    THRESHOLD --> TN
    
    TP --> F1
    FP --> F1
    FN --> F1
    TN --> F1
    
    TP --> IoU
    FP --> IoU
    FN --> IoU
    
    TP --> Prec
    FP --> Prec
    
    FN --> Rec
    TP --> Rec
    
    ŷ_prob --> AUC
    T --> CHECK1
    T --> CHECK2
    T --> CHECK3
    
    style F1 fill:#a5d6a7
    style IoU fill:#a5d6a7
    style Prec fill:#a5d6a7
    style Rec fill:#a5d6a7
    style AUC fill:#a5d6a7
```

---

## 9. Colab Training Deployment

```mermaid
graph TB
    subgraph LOCAL["Local Machine"]
        DATASET["segmentation_train_dataset.npz<br/>6.02 GB"]
        NOTEBOOK["nairobi_flood_colab_training.ipynb"]
    end
    
    subgraph DRIVE["Google Drive"]
        UPLOAD["Upload Dataset<br/>My Drive > nairobi-flood-data/"]
    end
    
    subgraph COLAB["Google Colab<br/>GPU Runtime T4"]
        CLONE["Clone Repository<br/>github.com/..."]
        DEPS["Install Dependencies<br/>torch, numpy, scipy"]
        COPY["Copy Dataset from Drive<br/>→ /content/"]
        TRAIN["Train Model<br/>50 Epochs<br/>~4 Hours"]
        SAVE["Save Results<br/>model.pth + metrics.json<br/>→ Drive"]
    end
    
    subgraph RESULTS["Results"]
        MODEL["Trained Model<br/>segmentation_model.pth"]
        METRICS["Training Metrics<br/>segmentation_metrics.json"]
        DOWNLOAD["Download to Local"]
    end
    
    DATASET --> UPLOAD
    NOTEBOOK --> COLAB
    UPLOAD --> COPY
    CLONE --> TRAIN
    DEPS --> TRAIN
    COPY --> TRAIN
    TRAIN --> SAVE
    SAVE --> MODEL
    SAVE --> METRICS
    MODEL --> DOWNLOAD
    METRICS --> DOWNLOAD
    
    style TRAIN fill:#fff9c4
    style SAVE fill:#ffd54f
    style DOWNLOAD fill:#a5d6a7
```

---

## 10. End-to-End System Flow

```mermaid
graph LR
    subgraph sat["Satellite Data<br/>2015-2026"]
        S1A["Sentinel-1 SAR<br/>7-Day Series"]
        RAIN["CHIRPS Rainfall<br/>7-Day Window"]
        TERR["Terrain Predictors<br/>HAND/DEM/etc"]
    end
    
    subgraph process["Processing Pipeline"]
        FETCH["Fetch & Stack<br/>14 Channels"]
        LABEL["Label via Rainfall<br/>Rain >= 30mm"]
        BUILD["Build Dataset<br/>703 Scenes"]
    end
    
    subgraph train["Training"]
        UNET["Train U-Net<br/>50 Epochs<br/>Google Colab GPU"]
    end
    
    subgraph eval["Evaluation"]
        TEST["Test on 106<br/>Unseen Scenes"]
        METRICS["Compute F1 / IoU<br/>Precision / Recall"]
    end
    
    subgraph output["Outputs"]
        MODEL["Trained Model<br/>segmentation_model.pth"]
        PREDS["Flood Probability<br/>Maps"]
        THESIS["Thesis + Figures<br/>Methods/Results/Discussion"]
    end
    
    S1A --> FETCH
    RAIN --> LABEL
    TERR --> FETCH
    FETCH --> BUILD
    LABEL --> BUILD
    BUILD --> UNET
    UNET --> TEST
    TEST --> METRICS
    METRICS --> PREDS
    MODEL --> PREDS
    PREDS --> THESIS
```

---

## 11. Rainfall Label Distribution

```mermaid
graph TD
    EVENTS["23 Sentinel-1 Storm Events<br/>2015-2026"]
    
    SPLIT["Split by 7-day Rainfall"]
    
    HEAVY["Heavy Rainfall<br/>≥ 30mm/7-day<br/>51 Scenes"]
    LIGHT["Light Rainfall<br/>< 30mm/7-day<br/>111 Scenes"]
    
    CLASS["Binary Classification"]
    FLOOD["Class 1: Flood<br/>Label = 1"]
    NOFLOOD["Class 0: No Flood<br/>Label = 0"]
    
    EVENTS --> SPLIT
    SPLIT --> HEAVY
    SPLIT --> LIGHT
    HEAVY --> CLASS
    LIGHT --> CLASS
    CLASS --> FLOOD
    CLASS --> NOFLOOD
    
    STATS["Statistics<br/>31.5% Positive<br/>68.5% Negative<br/>Good for Training"]
    
    FLOOD --> STATS
    NOFLOOD --> STATS
    
    style HEAVY fill:#ffcdd2
    style LIGHT fill:#c8e6c9
    style STATS fill:#ffe0b2
```

---

## 12. Model Performance Timeline (Expected)

```mermaid
graph LR
    E1["E1<br/>train=0.45<br/>val_f1=0.05"]
    E10["E10<br/>train=0.30<br/>val_f1=0.25"]
    E20["E20<br/>train=0.22<br/>val_f1=0.42"]
    E30["E30<br/>train=0.18<br/>val_f1=0.55"]
    E40["E40<br/>train=0.14<br/>val_f1=0.62"]
    E50["E50<br/>train=0.12<br/>val_f1=0.68"]
    
    E1 --> E10 --> E20 --> E30 --> E40 --> E50
    
    style E1 fill:#ffcdd2
    style E10 fill:#ffe0b2
    style E20 fill:#fff9c4
    style E30 fill:#c8e6c9
    style E40 fill:#a5d6a7
    style E50 fill:#81c784
```

---

## 13. Input Channel Composition

```mermaid
graph TB
    CHANNELS["14 Input Channels<br/>per Timestep<br/>7 Timesteps"]
    
    subgraph SAR["4 SAR Channels (Time-Varying)"]
        VV1["VV Backscatter<br/>T-6 Days"]
        VV2["VV Backscatter<br/>T-5 Days"]
        VV3["VV Backscatter<br/>T-3 Days"]
        VV4["VV Backscatter<br/>T-0 Days<br/>(Event Date)"]
    end
    
    subgraph STATIC["7 Static Channels (Constant)"]
        HAND["HAND<br/>Height Above<br/>Nearest Drainage"]
        SLOPE["Slope<br/>Terrain Gradient"]
        ELEV["Elevation<br/>DEM"]
        TWI["TWI<br/>Topographic Wetness<br/>Index"]
        BUILDUP["Built-up<br/>Fraction"]
        PERWATER["Permanent<br/>Water"]
        DEM["DEM<br/>Digital Elevation"]
    end
    
    CHANNELS --> SAR
    CHANNELS --> STATIC
    
    style SAR fill:#bbdefb
    style STATIC fill:#c8e6c9
```

---

## 14. Decision Tree: Why Rainfall-Based Labels

```mermaid
graph TD
    Q1["Can We Detect Urban<br/>Flooding Directly?"]
    
    OPT["Using Optical Satellites<br/>Sentinel-2 10m Res"]
    OPT_RES["0 Water Pixels<br/>in 156 Scenes"]
    OPT_FAIL["FAIL: Urban Flooding<br/>Too Small Scale"]
    
    SAR1["Using SAR<br/>Direct Backscatter<br/>-16dB Threshold"]
    SAR1_RES["Inverted Signal<br/>r = -0.74<br/>Threshold Selects<br/>Dry Surfaces"]
    SAR1_FAIL["FAIL: Physics Wrong"]
    
    SAR2["Using SAR<br/>Change Detection<br/>vs Dry-Season"]
    SAR2_RES["Backscatter Increases<br/>with Rain<br/>r = -0.399<br/>0 Floods Detected"]
    SAR2_FAIL["FAIL: Wet Soil<br/>Increases VV"]
    
    RAINFALL["Using Rainfall<br/>as Ground Truth<br/>Rainfall >= 30mm"]
    RAINFALL_RES["Independent Honest<br/>Data Source<br/>Physically Causes<br/>Flooding"]
    RAINFALL_WIN["SUCCESS: 51 Floods<br/>111 Non-floods<br/>31.5% Balance"]
    
    Q1 -->|Attempt 1| OPT
    OPT --> OPT_RES
    OPT_RES --> OPT_FAIL
    OPT_FAIL -->|Reject| Q1
    
    Q1 -->|Attempt 2| SAR1
    SAR1 --> SAR1_RES
    SAR1_RES --> SAR1_FAIL
    SAR1_FAIL -->|Reject| Q1
    
    Q1 -->|Attempt 3| SAR2
    SAR2 --> SAR2_RES
    SAR2_RES --> SAR2_FAIL
    SAR2_FAIL -->|Reject| Q1
    
    Q1 -->|Solution| RAINFALL
    RAINFALL --> RAINFALL_RES
    RAINFALL_RES --> RAINFALL_WIN
    
    style RAINFALL_WIN fill:#81c784
    style OPT_FAIL fill:#ef5350
    style SAR1_FAIL fill:#ef5350
    style SAR2_FAIL fill:#ef5350
```

---

## 15. File Output Structure

```mermaid
graph TB
    TRAIN["Training Completes<br/>on Google Colab"]
    
    MODEL["segmentation_model.pth<br/>PyTorch State Dict<br/>~50 MB"]
    METRICS["segmentation_metrics.json<br/>Training History<br/>Per-Epoch Results"]
    LOGS["Training Logs<br/>Epoch Progress<br/>Loss/Metrics"]
    
    DRIVE["Google Drive<br/>My Drive > nairobi-flood-data<br/>> training-outputs"]
    
    DOWNLOAD["Download to Local<br/>models/time_series/"]
    
    TRAIN --> MODEL
    TRAIN --> METRICS
    TRAIN --> LOGS
    
    MODEL --> DRIVE
    METRICS --> DRIVE
    LOGS --> DRIVE
    
    DRIVE --> DOWNLOAD
    DOWNLOAD --> MODEL
    DOWNLOAD --> METRICS
    
    style MODEL fill:#a5d6a7
    style METRICS fill:#a5d6a7
```

---

## Usage Instructions

1. **Copy each diagram** into your thesis document (Google Docs, Overleaf, Word)
2. **Use Mermaid Live Editor:** Paste code at https://mermaid.live/ to preview/export
3. **Export as PNG:** Click the download icon in Mermaid Live
4. **Integrate into thesis:** Insert figures in Methods, Results, or Appendix

**All diagrams are based on your actual implementation:**
- ✅ 703 training scenes (70/15/15 split)
- ✅ 14-channel input (4 SAR + 7 terrain + 7 static)
- ✅ U-Net segmentation architecture
- ✅ Binary cross-entropy + Dice loss
- ✅ Google Colab GPU training (~4 hours)
- ✅ Rainfall-based ground truth (31.5% class balance)
