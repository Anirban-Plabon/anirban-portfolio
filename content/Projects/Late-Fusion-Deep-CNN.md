## Late Fusion Deep CNN for Brain Tumor Classification

### 📊 Project Overview

**Type:** Research Project | Academic Publication  
**Duration:** January 2022 - August 2023  
**Status:** ✅ Published in IEEE Xplore  
**Team Size:** 6 researchers  
**Role:** Lead Researcher & First Author
**Paper Description:** [[Late Fusion]]

---

### 🎯 Problem Statement

Brain tumors are among the most severe forms of brain disease, with over 25,000 newly diagnosed cases annually in the US alone. Accurate classification of brain tumors (Higher-Grade Glioma vs. Lower-Grade Glioma) is crucial for:

- **Clinical Diagnosis** - Determining appropriate treatment plans
- **Prognosis** - Predicting patient outcomes
- **Treatment Decisions** - Guiding surgical and therapeutic interventions

**Challenge:** Multi-parametric MRI sequences (T1, T1CE, T2, FLAIR) each provide unique information about tumors, but effectively combining these modalities for classification was a significant challenge.

---

### 💡 Solution Approach

#### **Novel Late Fusion Architecture**

We developed a custom CNN architecture that:

1. **Processes Each Modality Independently**
   - Four separate Deep Feature Extraction Networks (DFEN)
   - Each DFEN tailored to extract features from specific MRI sequences
   - Preserves unique characteristics of each modality

2. **Late Fusion Strategy**
   - Features combined at the classification stage (not early fusion)
   - Concatenation of flattened layers (147,456 total features)
   - Fully connected layers for final classification

3. **Deep Feature Extraction Network (DFEN)**
   - 4 Conv2D layers with ReLU activation
   - 2 Average-pooling layers + 2 Max-pooling layers
   - Dropout layer for regularization
   - Outputs 36,864 features per modality

---

### 🏗️ Technical Architecture

#### **Input Processing**
```
MRI Sequences (4 modalities):
├── T1-weighted (T1)          → Anatomical details
├── T1CE (Contrast Enhanced)  → Blood-brain barrier permeability
├── T2-weighted (T2)          → Fluid visualization
└── FLAIR                     → Lesion enhancement

Each: 240 × 240 × 155 (H × W × Slices)
```

#### **DFEN Architecture per Modality**
```
Input: 224 × 224 × 1
    ↓
Conv2D (32 filters, 4×4) → 221 × 221 × 32
    ↓
AvgPooling → 110 × 110 × 32
    ↓
Conv2D (64 filters, 4×4) → 107 × 107 × 64
    ↓
AvgPooling → 53 × 53 × 64
    ↓
Conv2D (128 filters, 4×4) → 50 × 50 × 128
    ↓
MaxPooling → 25 × 25 × 128
    ↓
Conv2D (64 filters, 4×4) → 24 × 24 × 64
    ↓
MaxPooling → 12 × 12 × 64
    ↓
Dropout (0.25)
    ↓
Flatten → 36,864 features
```

#### **Late Fusion & Classification**
```
4 × Flatten Layers (36,864 each)
    ↓
Concatenate → 147,456 features
    ↓
Dense (32 nodes, ReLU)
    ↓
Dropout (0.25)
    ↓
Dense (2 nodes, Softmax) → [HGG, LGG]
```

---

### 📊 Dataset Details

**Dataset:** BraTS 2019 (Brain Tumor Segmentation Challenge)

| Class | Patients | Images per Patient | Total Slices |
|-------|----------|-------------------|--------------|
| HGG (Higher-Grade Glioma) | 262 | 4 sequences × 155 slices | 162,040 |
| LGG (Lower-Grade Glioma) | 199 | 4 sequences × 155 slices | 123,220 |
| **Total** | **461** | **620 images** | **285,260** |

**MRI Sequences:**
- T1-weighted (T1)
- T1-weighted with Contrast Enhancement (T1CE)
- T2-weighted (T2)
- Fluid-Attenuated Inversion Recovery (FLAIR)

**Image Resolution:** 240 × 240 pixels per slice

---

### 🔧 Implementation Details

#### **Data Preprocessing Pipeline**

1. **Slice Selection**
   ```python
   # Middle slice selection with randomization
   pos_mid = floor(n_slices / 2)
   selected_slice = random.uniform(pos_mid - 7, pos_mid + 7)
   ```

2. **Normalization**
   - Min-Max normalization to [0, 1] range
   - Pixel intensity standardization

3. **Data Augmentation**
   - Cropping (random regions)
   - Horizontal/Vertical flipping
   - Rotation (±15 degrees)
   - Zooming (0.9-1.1 scale)

4. **Train-Validation-Test Split**
   - Training: 60% (276 patients)
   - Validation: 20% (92 patients)
   - Testing: 20% (93 patients)

---

#### **Training Configuration**

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Learning Rate** | 0.0001 |
| **Decay Rate** | 0.96 |
| **Decay Steps** | 100,000 |
| **Batch Size** | 32 |
| **Epochs** | 100 |
| **Loss Function** | Categorical Cross-Entropy |
| **Metrics** | Accuracy, Precision, Recall, F1-Score |

**Hardware:**
- GPU: Tesla P100-PCIE (16GB VRAM)
- CPU: Intel Xeon 2.00 GHz (dual-core)
- RAM: 13 GB
- Platform: Kaggle Notebook

---

### 📈 Results & Performance

#### **Classification Performance**

| Metric | HGG Class | LGG Class | Overall |
|--------|-----------|-----------|---------|
| **Precision** | 98% | 97% | 98% |
| **Recall** | 96% | 99% | 97% |
| **F1-Score** | 97% | 98% | 97% |
| **Accuracy** | - | - | **97%** |

#### **Confusion Matrix**
```
                Predicted
              HGG    LGG
Actual  HGG   119     2
        LGG     1   130
```

**Metrics:**
- True Positives (HGG): 119
- False Positives (HGG): 1
- True Negatives (LGG): 130
- False Negatives (LGG): 2
- **Misclassification Rate: 1.19%**

#### **ROC-AUC Score**
- **AUC: 99%** - Excellent discrimination between classes

---

### 🏆 Comparative Analysis

Our model outperformed several state-of-the-art architectures:

| Architecture | Accuracy | Precision | Recall | F1-Score | Parameters |
|--------------|----------|-----------|--------|----------|------------|
| VGG-19 | 84% | 83% | 84% | 82% | 143M |
| EfficientNetB0 | 86% | 86% | 84% | 88% | 5.3M |
| EfficientNetB7 | 90% | 90% | 91% | 89% | 66M |
| InceptionV3 | 91% | 90% | 92% | 88% | 23.8M |
| ResNet101 | 93% | 92% | 93% | 91% | 44.5M |
| **Proposed Model** | **97%** | **98%** | **96%** | **97%** | **~15M** |

**Key Advantages:**
- ✅ Highest accuracy among compared models
- ✅ Better precision and recall balance
- ✅ Efficient parameter usage
- ✅ Robust generalization (low overfitting)

---

### 🔬 Research Contributions

#### **1. Novel Architecture Design**
- First implementation of late fusion for multi-parametric brain MRI
- Custom DFEN optimized for medical imaging
- Balanced feature extraction across modalities

#### **2. Multi-Modal Integration**
- Effective combination of four MRI sequences
- Preservation of modality-specific features
- Improved classification through complementary information

#### **3. Clinical Applicability**
- High accuracy suitable for clinical decision support
- Feature visualization for interpretability
- Potential for real-time diagnostic assistance

#### **4. Methodological Innovation**
- Comprehensive data augmentation strategy
- Optimized training pipeline
- Reproducible experimental setup

---

### 📊 Feature Visualization

**Generated Feature Maps:**
- Visualized 25 × 25 feature maps from final Conv2D layer
- Demonstrated activation patterns for each MRI sequence
- Highlighted tumor-relevant regions
- Provided interpretability for clinical use

**Key Insights:**
- T1CE shows strong activation in enhanced tumor regions
- FLAIR highlights edema and fluid accumulation
- T1 and T2 provide complementary structural information
- Late fusion captures all relevant features effectively

---

### 💻 Technologies & Tools

#### **Deep Learning Frameworks**
- **TensorFlow** 2.x
- **Keras** (high-level API)
- **NumPy** for numerical computations
- **Pandas** for data management

#### **Image Processing**
- **OpenCV** for image manipulation
- **PIL** for format handling
- **nibabel** for NIFTI file processing
- **scikit-image** for preprocessing

#### **Visualization**
- **Matplotlib** for plotting
- **Seaborn** for statistical visualization
- **TensorBoard** for training monitoring

#### **Development Environment**
- **Jupyter Notebook** for experimentation
- **Google Colab** for GPU access
- **Kaggle** for dataset and compute resources
- **Git/GitHub** for version control

---

### 📄 Publication Details

**Title:** *A Late Fusion Deep CNN Model for the Classification of Brain Tumors from Multi-Parametric MRI Images*

**Authors:** Anirban Barai, Md. Farukuzzaman Faruk, Shakil Mahmud Shuvo, Azmain Yakin Srizon, SM Mahedy Hasan, Abu Sayeed

**Conference:** 2023 International Conference on Next-Generation Computing, IoT and Machine Learning (NCIM)

**Date:** June 16-17, 2023

**Location:** Gazipur-1707, Bangladesh

**Publisher:** IEEE

**Indexed:** IEEE Xplore (August 2023)

**DOI:** [Available on IEEE Xplore]

---

### 🔗 Resources

- **📄 Full Paper:** [IEEE Xplore](https://ieeexplore.ieee.org/)
- **💻 Code Repository:** [GitHub - Late-Fusion-Deep-CNN](https://github.com/Anirban-Plabon/Late-Fusion-Deep-CNN)
- **📊 Dataset:** [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/)
- **📈 Results:** [Kaggle Notebook](#)

---

### 🎯 Future Work

#### **Immediate Extensions**
1. **3D CNN Implementation**
   - Process full 3D volumes instead of 2D slices
   - Capture spatial relationships across slices
   - Potentially improve accuracy further

2. **Segmentation Model**
   - Extend classification to precise tumor boundary detection
   - Multi-class segmentation (tumor core, edema, enhancement)
   - Clinical utility for surgical planning

3. **Attention Mechanisms**
   - Integrate attention layers to focus on tumor regions
   - Improve interpretability
   - Reduce computational requirements

#### **Long-term Goals**
1. **Clinical Validation**
   - Prospective study with real patient data
   - Collaboration with radiologists
   - FDA approval process

2. **Real-time Deployment**
   - Web-based diagnostic tool
   - Integration with hospital PACS systems
   - Mobile application for accessibility

3. **Synthetic Data Generation**
   - Use GANs to augment limited datasets
   - Address class imbalance
   - Improve model robustness

---

### 🏆 Impact & Recognition

#### **Academic Impact**
- ✅ Published in IEEE-indexed conference
- ✅ Cited by researchers in medical imaging
- ✅ Presented at international conference
- ✅ Contributed to brain tumor classification literature

#### **Educational Impact**
- ✅ Open-source code for learning
- ✅ Detailed documentation for reproduction
- ✅ Reference implementation for students
- ✅ Inspiration for future research

#### **Practical Impact**
- ✅ Demonstrated feasibility of automated diagnosis
- ✅ Potential for clinical decision support
- ✅ Cost-effective diagnostic solution
- ✅ Reduced radiologist workload

---

### 👥 Team & Acknowledgments

**Research Team:**
- **Anirban Barai** - Lead Researcher, Architecture Design, Implementation
- **Md. Farukuzzaman Faruk** - Supervisor, Research Guidance
- **Shakil Mahmud Shuvo** - Documentation
- **Azmain Yakin Srizon** - Documentation
- **SM Mahedy Hasan** - Documentation
- **Abu Sayeed** - Documentation

**Special Thanks:**
- RUET CSE Department for resources
- BraTS Challenge organizers for dataset
- IEEE NCIM 2023 conference organizers

---

### 📚 Key Learnings

#### **Technical Learnings**
1. Multi-modal data fusion strategies
2. Medical image preprocessing techniques
3. CNN architecture design for specific domains
4. Effective training strategies for limited data

#### **Research Skills**
1. Literature review and gap identification
2. Experimental design and methodology
3. Result analysis and interpretation
4. Academic writing and presentation

#### **Collaboration**
1. Working in research teams
2. Peer review and feedback integration
3. Conference presentation skills
4. Academic networking

---

### 💡 Project Highlights

> **"This project represents the perfect intersection of deep learning and healthcare—using AI to save lives through early and accurate brain tumor diagnosis."**

**Key Statistics:**
- 📊 **97% Accuracy** achieved
- 🏆 **Published** in IEEE Xplore
- 💻 **147,456** features fused
- 🧠 **461** patients in dataset
- ⏱️ **18 months** project duration
- 📈 **4% improvement** over best baseline

---


