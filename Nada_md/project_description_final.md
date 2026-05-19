# Comprehensive Pre-Implementation Report

## Compression-Aware Multiscale Feature Fusion for Waste Object Detection and Classification

## 1. Project Summary
This project investigates how different visual feature representations can be combined and
compressed to improve waste object detection and classification. The main idea is to compare
classical hand-crafted features, pretrained deep learning features, and YOLO detection-specific
features, then test different fusion and dimensionality reduction strategies.
The project is not simply a comparison between YOLO, SVM, and Random Forest. It is a
representation engineering study. The goal is to understand which visual features are most
useful for waste recognition, how they can be fused, and how much the final representation can
be reduced without losing significant performance. This matches the enhanced project
description, which focuses on combining hand-crafted and deep features while minimizing
computational cost.
The small-object focus will be removed completely. The project will focus on general waste
object detection and classification across all object sizes.

## 2. Proposed Project Title

### Recommended Title
Compression-Aware Multiscale Feature Fusion for Waste Object Detection and Classification

**Alternative title:**
Multiscale Feature Fusion and Dimensionality Reduction for Waste Object Detection
The first title is stronger because it clearly shows the two main technical ideas:
1. Feature fusion
2. Dimensionality reduction / compression

## 3. Main Research Question

**The main research question is:**
How can classical visual features, pretrained deep neural features, and YOLO detection
features be fused and compressed to improve waste-object classification accuracy while
reducing feature dimensionality and computational cost?
This question is clear, realistic, and directly connected to Computer Vision.

## 4. Project Objectives

**The project has five main objectives:**
Objective                        Explanation
Compare individual feature       Test classical, deep, and YOLO-based features separately
types
Study feature fusion             Compare early fusion, late fusion, attention fusion, and YOLO
hybrid fusion
Apply dimensionality             Use PCA, autoencoder compression, and feature selection
reduction
Evaluate accuracy-efficiency Compare accuracy, F1-score, mAP, speed, feature size, and
trade-off                    model size
Identify the best practical      Recommend the most accurate and efficient representation for
configuration                    waste detection/classification

## 5. Dataset
The project uses the Kaggle Trash Detection Image Dataset, which contains 4,142 images with
bounding box annotations in YOLO format. The original dataset includes seven classes:
Cardboard, Glass, Metal, Paper, Plastic, Trash, and Biodegradable. In the proposed project,
Biodegradable is merged with Trash, producing a final six-class taxonomy.

### Final Class List
Original Class    Final Class
Cardboard         Cardboard
Glass             Glass
Metal             Metal
Paper             Paper
Plastic           Plastic
Trash             Trash
Biodegradable Trash
The final classification problem therefore contains six material categories.

## 6. Dataset Split

**The dataset will be divided using stratified sampling:**
Split       Percentage Approximate Images
Training    70%             ~2,900
Validation 15%              ~620
Test        15%             ~620
The split must be done at the image level, not object level. This prevents objects from the same
image appearing in both training and test sets.
Stratification is important because some classes, especially Metal, may appear less frequently
than others. The split should preserve class distribution as much as possible.

## 7. Important Methodological Clarification

**This project contains two related but different tasks:**

### Task A: Object Detection

**Object detection means the system outputs:**
bounding box + class label + confidence score

**This applies mainly to:**
1. YOLOv8-nano baseline
2. YOLO hybrid model
Detection metrics such as mAP@0.5 and mAP@0.5:0.95 are valid here.

### Task B: Object Crop Classification
Object crop classification means the object location is already known, and the model only
predicts the material class.

**This applies to:**
1. Classical feature classifier
2. Deep feature classifier
3. Early fusion models
4. PCA fusion
5. Autoencoder fusion
6. Feature-selection fusion
7. Late fusion
8. Attention fusion
These models do not perform full detection by themselves. They classify cropped object regions.

**So their main metrics should be:**
accuracy, macro-F1, per-class F1, feature dimension, extraction time, and classification
time
This distinction is very important. It makes the methodology technically correct.

## 8. General Pipeline

**The full project pipeline will follow these stages:**
Stage 1: Dataset Preparation

1.     Load images and YOLO annotation files.
2.     Convert YOLO labels into bounding box coordinates.
3.     Merge Biodegradable into Trash.
4.     Split dataset into train, validation, and test sets.
5.     Crop object regions from images using ground-truth boxes.
6.     Resize crops when needed, especially for EfficientNet.

Stage 2: YOLO Training
A YOLOv8-nano detector will be trained on the six-class dataset.

**YOLO will be used for two purposes:**
1. As a detection baseline.
2. As a region proposal/localization module for hybrid experiments.

Stage 3: Feature Extraction
For each object crop, three types of features will be extracted:
1. Classical hand-crafted features
2. Deep EfficientNetB0 features
3. YOLO Feature Pyramid Network features

Stage 4: Fusion and Compression

**Different fusion strategies will be tested:**
1. Early feature concatenation
2. PCA dimensionality reduction
3. Autoencoder compression
4. Feature selection
5. Late probability fusion
6. Attention-based fusion
7. YOLO hybrid fusion

Stage 5: Evaluation
The project will evaluate both detection and classification performance.
Detection models will use mAP-based metrics.
Feature/classification models will use accuracy, macro-F1, per-class F1, time, and feature size.

## 9. Feature Extraction Design

### 9.1 Classical Feature Branch
The classical feature branch extracts interpretable visual properties from object crops.
Expected feature size: 252 dimensions
Feature Type                     Dimensions Purpose
Edge features                    5             Capture boundary and gradient information
Shape features                   19            Describe object geometry
LBP texture features             119           Capture local texture patterns
GLCM texture features            48            Capture texture regularity and direction
HSV color histogram              24            Capture color distribution
Spatial pyramid features         28            Preserve coarse spatial layout
Lab perceptual color features 7                Represent human-perceived color differences
Material-specific cues           2             Detect highlight/metallic-like properties
Total                            252           Full classical vector
This branch is useful because waste categories often depend on visible material properties such
as color, texture, reflectiveness, and shape.

### 9.2 Deep Hierarchical Feature Branch
The deep feature branch uses a pretrained EfficientNetB0 model.
The model is used as a feature extractor, not as a fully fine-tuned classifier.

Feature Level            Dimensions Meaning
Early-layer features 256                   Edges, corners, colors, low-level texture
Mid-layer features       512               Material surfaces, local structures
Late-layer features      1,280             High-level semantic appearance
Total deep features 2,048                  Full EfficientNet representation
The early, mid, and late features capture different levels of visual information.

### 9.3 YOLO Detection Feature Branch
YOLOv8-nano produces multiscale feature maps through its Feature Pyramid Network.
YOLO Feature Map               Dimensions Role
P3                             256              Higher-resolution features
P4                             256              Medium-scale features
P5                             256              Lower-resolution semantic features
Total YOLO FPN features 768                     Detection-specific representation
These features are useful because they are directly optimized for object detection.

## 10. Experiments to Be Conducted
The project should include eleven experimental configurations.

### Experiment Group 1: Baseline Experiments
These experiments establish the performance of each representation type before fusion.
Experiment Name                             Feature Source             Model           Evaluation Type

### E1           YOLOv8-nano                    Full image YOLO            YOLO detector Detection
             baseline                       features

### E2           Classical-only baseline 252 classical features            Random          Classification
Forest

### E3           Deep-only baseline             EfficientNet late          SVM             Classification
features

### E1: YOLOv8-Nano Baseline
This experiment trains YOLOv8-nano on the waste dataset.

**Purpose:**
Establish the main object detection baseline.

**Outputs:**
- Bounding boxes
- Class labels
- Confidence scores

**Metrics:**
- mAP@0.5
- mAP@0.5:0.95
- Per-class AP
- FPS
- Model size

### E2: Classical-Only Baseline
This experiment extracts 252-dimensional classical features from each object crop and trains a
Random Forest classifier.

**Purpose:**
Test how well interpretable visual features classify waste materials.

**Metrics:**
- Accuracy
- Macro-F1
- Per-class F1
- Feature extraction time

- Classification time

### E3: Deep-Only Baseline
This experiment extracts EfficientNetB0 late-layer features and trains an SVM classifier.

**Purpose:**
Test how well pretrained deep semantic features classify waste objects.

**Metrics:**
- Accuracy
- Macro-F1
- Per-class F1
- Feature extraction time
- Classification time

### Experiment Group 2: Early Fusion Experiments
Early fusion means features are combined first, then one classifier is trained.
Experime Name                      Features Used            Reduction Method Classifier
nt

### E4          Raw early fusion Classical + EfficientNet       None                  Random Forest
early + mid + late

### E5          PCA fusion             Same fused vector        PCA, 95% variance SVM

### E6          Autoencoder            Same fused vector        256-dimensional       Neural classifier
            fusion                                          bottleneck            / SVM

### E7          Feature-               Same fused vector        Top 200 features      Random Forest
selection fusion

**The full early-fusion vector is:**
Feature Source      Dimensions
Classical features 252
EfficientNet early 256
EfficientNet mid    512
EfficientNet late   1,280
Total               2,300

### E4: Raw Early Fusion
The 2,300-dimensional fused feature vector is used directly.

**Purpose:**
Test whether combining all available classical and deep features improves classification
performance.

**Main question:**
Does raw feature concatenation outperform individual feature branches?

### E5: PCA Fusion
The 2,300-dimensional vector is compressed using PCA while retaining 95% of variance.

**Purpose:**
Test whether linear dimensionality reduction can reduce feature size without large
accuracy loss.

**Expected output:**
- Reduced feature vector, probably around 400–600 dimensions
- Classification results after compression

### E6: Autoencoder Fusion
A shallow autoencoder compresses the 2,300-dimensional vector into a 256-dimensional
bottleneck.

**Suggested structure:**
Input 2300 → Dense 512 → Dense 256 → Dense 512 → Output 2300
The 256-dimensional bottleneck is then used for classification.

**Purpose:**
Test whether nonlinear compression preserves information better than PCA.

### E7: Feature-Selection Fusion
A Random Forest is used to rank feature importance.
The top 200 features are selected and used for classification.

**Purpose:**
Test whether only a small subset of the fused features is enough for strong classification
performance.
This experiment is useful because it may reveal which feature groups are most informative.

### Experiment Group 3: Late Fusion Experiments
Late fusion means separate classifiers are trained first, then their predictions are combined.
Experiment Name                  Models Combined                      Fusion Method

### E8            Average voting     Classical classifier + deep          Equal probability averaging
classifier

### E9            Weighted voting Classical classifier + deep             Validation-optimized
                              classifier                              weights

### E8: Average Voting
The classical classifier and deep classifier each produce class probabilities.

**The final prediction is:**
average of both probability vectors

**Purpose:**
Test whether simple decision-level fusion improves performance.

### E9: Weighted Voting
The classical and deep probabilities are combined using optimized weights.

**Example:**
Final prediction = 0.4 classical + 0.6 deep
The best weights are selected using the validation set.

**Purpose:**
Test whether one representation should be trusted more than another.
This may show that deep features are generally stronger, but classical features still add useful
material-specific information.

### Experiment Group 4: Attention-Based Fusion
Experiment Name                  Features Used                           Method

### E10           Attention fusion Classical + EfficientNet late features Learned gating network

### E10: Attention Fusion
This experiment trains a lightweight neural network that learns how much attention to give to
each feature branch.

**General architecture:**
1. Project classical features from 252 dimensions to 256 dimensions.
2. Project EfficientNet late features from 1,280 dimensions to 256 dimensions.
3. Concatenate both projected vectors.
4. Learn attention weights for classical and deep features.
5. Produce one fused 256-dimensional vector.
6. Classify the object.

**Purpose:**
Test whether the system can dynamically choose which representation is more useful for
each object.

**Example interpretation:**
- Cardboard may rely more on color and texture.
- Glass may rely more on shape, transparency, and deep visual patterns.

- Metal may rely on reflectiveness and deep features.
This experiment is advanced and gives the project a strong research contribution.

### Experiment Group 5: YOLO Hybrid Experiment
Experimen Name                  Features Used            Method              Evaluation
t

### E11            YOLO             YOLO FPN + classical     Fusion/re-scoring   Detection +
               hybrid           features                 module              classification

### E11: YOLO Hybrid
This experiment combines YOLO’s internal detection features with classical hand-crafted
features from the same detected region.

**Purpose:**
Test whether YOLO detection features become stronger when combined with explicit
color, texture, and material cues.

**Suggested implementation:**
1. Run YOLO on the image.
2. Extract predicted bounding boxes.
3. Crop each detected object region.
4. Extract classical features from the crop.
5. Extract YOLO FPN features corresponding to the detected region.
6. Concatenate YOLO FPN features with classical features.
7. Use a classifier or re-scoring module to refine the class prediction.

**Final output:**
- YOLO bounding box
- Refined class prediction
- Refined confidence score
Because this system still outputs bounding boxes, class labels, and confidence scores, it can be
evaluated using detection metrics.

## 11. Final Experiment Count
The project contains 11 experiments.
Group                  Experiments Count
Baselines              E1, E2, E3      3
Early fusion           E4, E5, E6, E7 4
Late fusion            E8, E9          2
Attention fusion E10                   1
YOLO hybrid            E11             1
Total                                  11
So in the report, do not write “thirteen experiments.”

**Use:**
eleven experimental configurations

**or:**
all experimental configurations

## 12. Evaluation Protocol
The evaluation should be divided into two tracks.

### Track A: Detection Evaluation

**Used for:**
- E1: YOLOv8-nano baseline
- E11: YOLO hybrid
        Metric               Meaning
        mAP@0.5              Detection performance at IoU threshold 0.5
        mAP@0.5:0.95         Stricter detection/localization metric
        Per-class AP         Detection performance for each waste class

      Precision            Correctness of positive detections
      Recall               Ability to find existing objects
      FPS                  Real-time speed
Model footprint Deployment size

### Track B: Classification / Representation Evaluation

**Used for:**
- E2: Classical-only
- E3: Deep-only
- E4: Raw early fusion
- E5: PCA fusion
- E6: Autoencoder fusion
- E7: Feature-selection fusion
- E8: Average voting
- E9: Weighted voting
- E10: Attention fusion
      Metric                       Meaning
      Accuracy                     Overall classification correctness
      Macro-F1                     Balanced performance across all classes
      Per-class F1                 Performance for each waste material
      Confusion matrix             Shows which classes are confused
      Feature dimension            Size of the representation
Feature extraction time Cost of producing the features
      Classification time          Cost of predicting the class
      Model size                   Storage requirement
Macro-F1 is especially important because the dataset may be imbalanced.

## 13. Controlled Variables
To ensure fair comparison, the following conditions should remain fixed:
Controlled Variable             Decision
Dataset split                   Same train/validation/test split for all experiments
Class mapping                   Same six-class taxonomy
Crop generation                 Same ground-truth crops for classification experiments
Image preprocessing             Same resizing and normalization rules
Classifier training data        Same training objects
Test set                        Same test objects/images
Random seed                     Fixed, for example 42
YOLO confidence threshold Fixed, for example 0.25
NMS IoU threshold               Fixed, for example 0.5

## 14. Recommended Implementation Order
To make the project manageable, implement the experiments in this order:

### Phase 1: Dataset and YOLO Baseline
1. Prepare dataset.
2. Merge classes.
3. Train YOLOv8-nano.
4. Evaluate YOLO detection performance.

**Output:**
YOLO baseline results.

### Phase 2: Object Crop Dataset
1. Extract object crops from ground-truth bounding boxes.
2. Save crop images and labels.
3. Prepare train/validation/test crop-level datasets.

**Output:**
Clean crop classification dataset.

### Phase 3: Baseline Feature Models
1. Extract classical features.
2. Train Random Forest.
3. Extract EfficientNet features.
4. Train SVM.
5. Compare classical-only and deep-only baselines.

**Output:**
Baseline representation results.

### Phase 4: Early Fusion and Compression
1. Concatenate classical + EfficientNet features.
2. Train raw fusion model.
3. Apply PCA.
4. Train PCA-fusion classifier.
5. Train autoencoder.
6. Extract bottleneck features.
7. Train autoencoder-fusion classifier.
8. Perform feature selection.
9. Train selected-feature classifier.

**Output:**
Accuracy vs feature-dimension comparison.

### Phase 5: Late Fusion
1. Generate probability outputs from classical and deep classifiers.
2. Apply average voting.
3. Optimize voting weights on validation set.
4. Evaluate on test set.

**Output:**
Decision-level fusion comparison.

### Phase 6: Attention Fusion
1. Build attention/gating network.
2. Train on classical + deep features.
3. Evaluate classification performance.
4. Analyze attention weights by class.

**Output:**
Interpretable learned fusion model.

### Phase 7: YOLO Hybrid
1. Extract YOLO FPN features.
2. Combine YOLO FPN features with classical crop features.
3. Train hybrid classifier/re-scoring module.
4. Evaluate detection/classification performance.

**Output:**
Final hybrid detection result.
This is the most technically challenging experiment, so it should be done after the main feature-
fusion experiments are working.

## 15. Expected Results
The expected findings should be written carefully, not as guaranteed results.

### Expected Finding 1: Fusion Improves Classification
Classical features may perform well on materials with distinctive color and texture, such as
cardboard and paper.
Deep features may perform better on visually complex materials such as glass, metal, and
plastic.
Therefore, fusion is expected to outperform either feature type alone.

### Expected Finding 2: PCA May Preserve Most Performance
PCA is expected to reduce the 2,300-dimensional fused vector significantly while preserving
most classification performance.
This would show that the full fused vector contains redundancy.

### Expected Finding 3: Autoencoder May Help at Strong
Compression
At very low dimensions, autoencoder compression may preserve nonlinear feature relationships
better than PCA.
This is useful if the final system is intended for deployment on limited hardware.

### Expected Finding 4: Attention Fusion May Be More
Interpretable
The attention model may learn different branch weights for different waste classes.

**For example:**
Class         Possible Dominant Features
Cardboard Classical color/texture
Paper         Classical texture/color
Glass         Deep visual patterns
Metal         Deep + reflectiveness cues
Plastic       Mixed classical/deep
Trash         Mixed features

### Expected Finding 5: YOLO Hybrid May Improve Class
Confidence
The YOLO hybrid model may improve classification confidence by adding explicit material cues
to YOLO’s detection features.
However, it may not greatly improve localization because bounding box prediction still comes
mainly from YOLO.

## 16. Main Risks and Solutions
Risk                            Explanation                        Solution
Too many experiments            Eleven experiments can be time-    Implement in phases and
                                consuming                          keep YOLO hybrid last
Confusion between               Classical/SVM models do not        Use two evaluation tracks
detection and classification    detect objects by themselves
YOLO FPN extraction may be Accessing internal YOLO features        Treat YOLO hybrid as
difficult                  requires extra coding                   advanced/final experiment
Dataset imbalance               Some waste classes may have        Use macro-F1 and class
                                fewer examples                     weights
Feature vector too large        2,300 dimensions may be            Use PCA, autoencoder, and
                                computationally heavy              feature selection
Overfitting                     Some models may memorize           Use validation set, test set,
                                crop-level features                and fixed split

## 17. Final Deliverables
By the end of the project, the final submission should include:
Deliverable                     Description
Dataset preprocessing code      Class merging, splitting, crop extraction
YOLO training results           Detection baseline
Feature extraction scripts      Classical, EfficientNet, YOLO FPN
Trained classifiers             RF, SVM, autoencoder, attention model
Experiment results table        All 11 experiments compared
Confusion matrices              For classification experiments
Accuracy vs dimension chart Shows compression trade-off
Speed comparison                Feature extraction and classification time
Final discussion                Best model and explanation
Final report/presentation       Academic project documentation

## 18. Suggested Final Results Table Format

Your final report should include a table like this:

| Exp. | Method | Feature Dim. | Accuracy | Macro-F1 | mAP@0.5 | FPS | Model Size |
|---|---|---:|---:|---:|---:|---:|---:|
| E1 | YOLOv8-nano | Native | — | — | — | — | — |
| E2 | Classical + RF | 252 | — | — | N/A | — | — |
| E3 | EfficientNet + SVM | 1280 | — | — | N/A | — | — |
| E4 | Raw early fusion | 2300 | — | — | N/A | — | — |
| E5 | PCA fusion | ~400-600 | — | — | N/A | — | — |
| E6 | Autoencoder fusion | 256 | — | — | N/A | — | — |
| E7 | Feature selection | 200 | — | — | N/A | — | — |
| E8 | Average voting | Separate | — | — | N/A | — | — |
| E9 | Weighted voting | Separate | — | — | N/A | — | — |
| E10 | Attention fusion | 256 | — | — | N/A | — | — |
| E11 | YOLO hybrid | 768 + 252 | — | — | — | — | — |

Use N/A where mAP does not apply.

## 19. Suggested Report Structure for Final Submission

When writing the final academic report, use this structure:

1. Abstract
2. Introduction
3. Problem Statement
4. Research Question
5. Dataset Description
6. Preprocessing
7. Feature Extraction Methods
8. Fusion Strategies
9. Dimensionality Reduction Methods
10. Experimental Design
11. Evaluation Metrics
12. Results
13. Discussion
14. Limitations
15. Conclusion
16. References

## 20. Final Feasibility Verdict

This project is viable and strong, as long as it is implemented with the correct separation between detection and classification.

The most important correction is:

YOLO performs localization and detection. Classical, deep, and fused features mainly evaluate object crop classification and representation quality.

The project should not claim that Random Forest, SVM, PCA, or attention fusion are full object detectors unless they are integrated into a full detection pipeline with bounding boxes and confidence scores.

The strongest final version is:

Train YOLOv8-nano as the detection baseline, then conduct a systematic representation study on waste object crops using classical features, EfficientNet features, fusion methods, and compression techniques. Finally, test whether YOLO’s detection features can be improved through a hybrid fusion model.
