

Comprehensive Pre-Implementation Report
Compression-Aware Multiscale Feature Fusion for Waste
Object Detection and Classification
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
object detection and classificationacross all object sizes.
## 2. Proposed Project Title
## Recommended Title
Compression-Aware Multiscale Feature Fusion for Waste Object Detection and Classification
Alternative title:
Multiscale Feature Fusion and Dimensionality Reduction for Waste Object Detection
The first title is stronger because it clearly shows the two main technical ideas:
Feature fusion 1.
Dimensionality reduction / compression 2.
## 3. Main Research Question
The main research question is:
How can classical visual features, pretrained deep neural features, and YOLO detection
features be fused and compressed to improve waste-object classification accuracy while
reducing feature dimensionality and computational cost?
This question is clear, realistic, and directly connected to Computer Vision.
## 4. Project Objectives
The project has five main objectives:
ObjectiveExplanation
Compare individual feature
types
Test classical, deep, and YOLO-based features separately
Study feature fusionCompare early fusion, late fusion, attention fusion, and YOLO
hybrid fusion
Apply dimensionality
reduction
Use PCA, autoencoder compression, and feature selection
Evaluate accuracy-efficiency
trade-off
Compare accuracy, F1-score, mAP, speed, feature size, and
model size
Identify the best practical
configuration
Recommend the most accurate and efficient representation for
waste detection/classification
## 5. Dataset
The project uses the Kaggle Trash Detection Image Dataset, which contains 4,142 imageswith
bounding box annotations in YOLO format. The original dataset includes seven classes:
Cardboard, Glass, Metal, Paper, Plastic, Trash, and Biodegradable. In the proposed project,
Biodegradable is merged with Trash, producing a final six-class taxonomy.
## Final Class List
Original ClassFinal Class
CardboardCardboard
GlassGlass
MetalMetal
PaperPaper
PlasticPlastic
TrashTrash
BiodegradableTrash
The final classification problem therefore contains six material categories.
## 6. Dataset Split
The dataset will be divided using stratified sampling:
SplitPercentageApproximate Images
## Training70%~2,900
## Validation15%~620
## Test15%~620
The split must be done at the image level, not object level. This prevents objects from the same
image appearing in both training and test sets.
Stratification is important because some classes, especially Metal, may appear less frequently
than others. The split should preserve class distribution as much as possible.
## 7. Important Methodological Clarification
This project contains two related but different tasks:
## Task A: Object Detection
Object detection means the system outputs:
bounding box + class label + confidence score
This applies mainly to:
YOLOv8-nano baseline 1.
YOLO hybrid model 2.
Detection metrics such as mAP@0.5and mAP@0.5:0.95are valid here.
## Task B: Object Crop Classification
Object crop classification means the object location is already known, and the model only
predicts the material class.
This applies to:
Classical feature classifier 1.
Deep feature classifier 2.
Early fusion models 3.
PCA fusion 4.
Autoencoder fusion 5.
Feature-selection fusion 6.
Late fusion 7.
Attention fusion 8.
These models do not perform full detection by themselves. They classify cropped object regions.
So their main metrics should be:
accuracy, macro-F1, per-class F1, feature dimension, extraction time, and classification
time
This distinction is very important. It makes the methodology technically correct.
## 8. General Pipeline
The full project pipeline will follow these stages:
## Stage 1: Dataset Preparation
Load images and YOLO annotation files. 1.
Convert YOLO labels into bounding box coordinates. 2.
Merge Biodegradable into Trash. 3.
Split dataset into train, validation, and test sets. 4.
Crop object regions from images using ground-truth boxes. 5.
Resize crops when needed, especially for EfficientNet. 6.
Stage 2: YOLO Training
A YOLOv8-nano detector will be trained on the six-class dataset.
YOLO will be used for two purposes:
As a detection baseline. 1.
As a region proposal/localization module for hybrid experiments. 2.
## Stage 3: Feature Extraction
For each object crop, three types of features will be extracted:
Classical hand-crafted features 1.
Deep EfficientNetB0 features 2.
YOLO Feature Pyramid Network features 3.
Stage 4: Fusion and Compression
Different fusion strategies will be tested:
Early feature concatenation 1.
PCA dimensionality reduction 2.
Autoencoder compression 3.
Feature selection 4.
Late probability fusion 5.
Attention-based fusion 6.
YOLO hybrid fusion 7.
## Stage 5: Evaluation
The project will evaluate both detection and classification performance.
Detection models will use mAP-based metrics.
Feature/classification models will use accuracy, macro-F1, per-class F1, time, and feature size.
## 9. Feature Extraction Design
## 9.1 Classical Feature Branch
The classical feature branch extracts interpretable visual properties from object crops.
Expected feature size: 252 dimensions
Feature TypeDimensionsPurpose
Edge features5Capture boundary and gradient information
Shape features19Describe object geometry
LBP texture features119Capture local texture patterns
GLCM texture features48Capture texture regularity and direction
HSV color histogram24Capture color distribution
Spatial pyramid features28Preserve coarse spatial layout
Lab perceptual color features7Represent human-perceived color differences
Material-specific cues2Detect highlight/metallic-like properties
Total252Full classical vector
This branch is useful because waste categories often depend on visible material properties such
as color, texture, reflectiveness, and shape.
## 9.2 Deep Hierarchical Feature Branch
The deep feature branch uses a pretrained EfficientNetB0 model.
The model is used as a feature extractor, not as a fully fine-tuned classifier.
Feature LevelDimensionsMeaning
Early-layer features256Edges, corners, colors, low-level texture
Mid-layer features512Material surfaces, local structures
Late-layer features1,280High-level semantic appearance
Total deep features2,048Full EfficientNet representation
The early, mid, and late features capture different levels of visual information.
9.3 YOLO Detection Feature Branch
YOLOv8-nano produces multiscale feature maps through its Feature Pyramid Network.
YOLO Feature MapDimensionsRole
P3256Higher-resolution features
P4256Medium-scale features
P5256Lower-resolution semantic features
Total YOLO FPN features768Detection-specific representation
These features are useful because they are directly optimized for object detection.
- Experiments to Be Conducted
The project should include eleven experimental configurations.
## Experiment Group 1: Baseline Experiments
These experiments establish the performance of each representation type before fusion.
ExperimentNameFeature SourceModelEvaluation Type
E1YOLOv8-nano
baseline
Full image YOLO
features
YOLO detectorDetection
E2Classical-only baseline252 classical featuresRandom
## Forest
## Classification
E3Deep-only baselineEfficientNet late
features
SVMClassification
E1: YOLOv8-Nano Baseline
This experiment trains YOLOv8-nano on the waste dataset.
## Purpose:
Establish the main object detection baseline.
## Outputs:
Bounding boxes •
Class labels •
Confidence scores •
## Metrics:
mAP@0.5 •
mAP@0.5:0.95 •
Per-class AP •
## FPS •
Model size •
E2: Classical-Only Baseline
This experiment extracts 252-dimensional classical features from each object crop and trains a
Random Forest classifier.
## Purpose:
Test how well interpretable visual features classify waste materials.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
E3: Deep-Only Baseline
This experiment extracts EfficientNetB0 late-layer features and trains an SVM classifier.
## Purpose:
Test how well pretrained deep semantic features classify waste objects.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
## Experiment Group 2: Early Fusion Experiments
Early fusion means features are combined first, then one classifier is trained.
## Experime
nt
NameFeatures UsedReduction MethodClassifier
E4Raw early fusionClassical + EfficientNet
early + mid + late
NoneRandom Forest
E5PCA fusionSame fused vectorPCA, 95% varianceSVM
E6Autoencoder
fusion
Same fused vector256-dimensional
bottleneck
Neural classifier
## / SVM
E7Feature-
selection fusion
Same fused vectorTop 200 featuresRandom Forest
The full early-fusion vector is:
Feature SourceDimensions
Classical features252
EfficientNet early256
EfficientNet mid512
EfficientNet late1,280
## Total2,300
## E4: Raw Early Fusion
The 2,300-dimensional fused feature vector is used directly.
## Purpose:
Test whether combining all available classical and deep features improves classification
performance.
Main question:
Does raw feature concatenation outperform individual feature branches?
E5: PCA Fusion
The 2,300-dimensional vector is compressed using PCA while retaining 95% of variance.
## Purpose:
Test whether linear dimensionality reduction can reduce feature size without large
accuracy loss.
Expected output:
Reduced feature vector, probably around 400–600 dimensions •
Classification results after compression •
## E6: Autoencoder Fusion
A shallow autoencoder compresses the 2,300-dimensional vector into a 256-dimensional
bottleneck.
Suggested structure:
## Input 2300 → Dense 512 → Dense 256 → Dense 512 → Output 2300
The 256-dimensional bottleneck is then used for classification.
## Purpose:
Test whether nonlinear compression preserves information better than PCA.
E7: Feature-Selection Fusion
A Random Forest is used to rank feature importance.
The top 200 features are selected and used for classification.
## Purpose:
Test whether only a small subset of the fused features is enough for strong classification
performance.
This experiment is useful because it may reveal which feature groups are most informative.
## Experiment Group 3: Late Fusion Experiments
Late fusion means separate classifiers are trained first, then their predictions are combined.
ExperimentNameModels CombinedFusion Method
E8Average votingClassical classifier + deep
classifier
Equal probability averaging
E9Weighted
voting
Classical classifier + deep
classifier
## Validation-optimized
weights
## E8: Average Voting
The classical classifier and deep classifier each produce class probabilities.
The final prediction is:
average of both probability vectors
## Purpose:
Test whether simple decision-level fusion improves performance.
## E9: Weighted Voting
The classical and deep probabilities are combined using optimized weights.
## Example:
Final prediction = 0.4 classical + 0.6 deep
The best weights are selected using the validation set.
## Purpose:
Test whether one representation should be trusted more than another.
This may show that deep features are generally stronger, but classical features still add useful
material-specific information.
Experiment Group 4: Attention-Based Fusion
ExperimentNameFeatures UsedMethod
E10Attention fusionClassical + EfficientNet late featuresLearned gating network
## E10: Attention Fusion
This experiment trains a lightweight neural network that learns how much attention to give to
each feature branch.
General architecture:
Project classical features from 252 dimensions to 256 dimensions. 1.
Project EfficientNet late features from 1,280 dimensions to 256 dimensions. 2.
Concatenate both projected vectors. 3.
Learn attention weights for classical and deep features. 4.
Produce one fused 256-dimensional vector. 5.
Classify the object. 6.
## Purpose:
Test whether the system can dynamically choose which representation is more useful for
each object.
Example interpretation:
Cardboard may rely more on color and texture. •
Glass may rely more on shape, transparency, and deep visual patterns. •
Metal may rely on reflectiveness and deep features. •
This experiment is advanced and gives the project a strong research contribution.
Experiment Group 5: YOLO Hybrid Experiment
## Experimen
t
NameFeatures UsedMethodEvaluation
## E11YOLO
hybrid
YOLO FPN + classical
features
## Fusion/re-scoring
module
## Detection +
classification
E11: YOLO Hybrid
This experiment combines YOLO’s internal detection features with classical hand-crafted
features from the same detected region.
## Purpose:
Test whether YOLO detection features become stronger when combined with explicit
color, texture, and material cues.
Suggested implementation:
Run YOLO on the image. 1.
Extract predicted bounding boxes. 2.
Crop each detected object region. 3.
Extract classical features from the crop. 4.
Extract YOLO FPN features corresponding to the detected region. 5.
Concatenate YOLO FPN features with classical features. 6.
Use a classifier or re-scoring module to refine the class prediction. 7.
Final output:
YOLO bounding box •
Refined class prediction •
Refined confidence score •
Because this system still outputs bounding boxes, class labels, and confidence scores, it can be
evaluated using detection metrics.
## 11. Final Experiment Count
The project contains 11 experiments.
GroupExperimentsCount
BaselinesE1, E2, E33
Early fusionE4, E5, E6, E74
Late fusionE8, E92
Attention fusionE101
YOLO hybridE111
## Total11
So in the report, do not write “thirteen experiments.”
## Use:
eleven experimental configurations
or:
all experimental configurations
## 12. Evaluation Protocol
The evaluation should be divided into two tracks.
## Track A: Detection Evaluation
Used for:
E1: YOLOv8-nano baseline •
E11: YOLO hybrid •
MetricMeaning
mAP@0.5Detection performance at IoU threshold 0.5
mAP@0.5:0.95Stricter detection/localization metric
Per-class APDetection performance for each waste class
PrecisionCorrectness of positive detections
RecallAbility to find existing objects
FPSReal-time speed
Model footprintDeployment size
## Track B: Classification / Representation Evaluation
Used for:
## E2: Classical-only •
## E3: Deep-only •
E4: Raw early fusion •
E5: PCA fusion •
E6: Autoencoder fusion •
E7: Feature-selection fusion •
E8: Average voting •
E9: Weighted voting •
E10: Attention fusion •
MetricMeaning
AccuracyOverall classification correctness
Macro-F1Balanced performance across all classes
Per-class F1Performance for each waste material
Confusion matrixShows which classes are confused
Feature dimensionSize of the representation
Feature extraction timeCost of producing the features
Classification timeCost of predicting the class
Model sizeStorage requirement
Macro-F1 is especially important because the dataset may be imbalanced.
## 13. Controlled Variables
To ensure fair comparison, the following conditions should remain fixed:
Controlled VariableDecision
Dataset splitSame train/validation/test split for all experiments
Class mappingSame six-class taxonomy
Crop generationSame ground-truth crops for classification experiments
Image preprocessingSame resizing and normalization rules
Classifier training dataSame training objects
Test setSame test objects/images
Random seedFixed, for example 42
YOLO confidence thresholdFixed, for example 0.25
NMS IoU thresholdFixed, for example 0.5
## 14. Recommended Implementation Order
To make the project manageable, implement the experiments in this order:
Phase 1: Dataset and YOLO Baseline
Prepare dataset. 1.
Merge classes. 2.
Train YOLOv8-nano. 3.
Evaluate YOLO detection performance. 4.
## Output:
YOLO baseline results.
## Phase 2: Object Crop Dataset
Extract object crops from ground-truth bounding boxes. 1.
Save crop images and labels. 2.
Prepare train/validation/test crop-level datasets. 3.
## Output:
Clean crop classification dataset.
## Phase 3: Baseline Feature Models
Extract classical features. 1.
## Train Random Forest. 2.
Extract EfficientNet features. 3.
Train SVM. 4.
Compare classical-only and deep-only baselines. 5.
## Output:
Baseline representation results.
Phase 4: Early Fusion and Compression
Concatenate classical + EfficientNet features. 1.
Train raw fusion model. 2.
Apply PCA. 3.
Train PCA-fusion classifier. 4.
Train autoencoder. 5.
Extract bottleneck features. 6.
Train autoencoder-fusion classifier. 7.
Perform feature selection. 8.
Train selected-feature classifier. 9.
## Output:
Accuracy vs feature-dimension comparison.
## Phase 5: Late Fusion
Generate probability outputs from classical and deep classifiers. 1.
Apply average voting. 2.
Optimize voting weights on validation set. 3.
Evaluate on test set. 4.
## Output:
Decision-level fusion comparison.
## Phase 6: Attention Fusion
Build attention/gating network. 1.
Train on classical + deep features. 2.
Evaluate classification performance. 3.
Analyze attention weights by class. 4.
## Output:
Interpretable learned fusion model.
Phase 7: YOLO Hybrid
Extract YOLO FPN features. 1.
Combine YOLO FPN features with classical crop features. 2.
Train hybrid classifier/re-scoring module. 3.
Evaluate detection/classification performance. 4.
## Output:
Final hybrid detection result.
This is the most technically challenging experiment, so it should be done after the main feature-
fusion experiments are working.
## 15. Expected Results
The expected findings should be written carefully, not as guaranteed results.
## Expected Finding 1: Fusion Improves Classification
Classical features may perform well on materials with distinctive color and texture, such as
cardboard and paper.
Deep features may perform better on visually complex materials such as glass, metal, and
plastic.
Therefore, fusion is expected to outperform either feature type alone.
Expected Finding 2: PCA May Preserve Most Performance
PCA is expected to reduce the 2,300-dimensional fused vector significantly while preserving
most classification performance.
This would show that the full fused vector contains redundancy.
Expected Finding 3: Autoencoder May Help at Strong
## Compression
At very low dimensions, autoencoder compression may preserve nonlinear feature relationships
better than PCA.
This is useful if the final system is intended for deployment on limited hardware.
## Expected Finding 4: Attention Fusion May Be More
## Interpretable
The attention model may learn different branch weights for different waste classes.
For example:
ClassPossible Dominant Features
CardboardClassical color/texture
PaperClassical texture/color
GlassDeep visual patterns
MetalDeep + reflectiveness cues
PlasticMixed classical/deep
TrashMixed features
Expected Finding 5: YOLO Hybrid May Improve Class
## Confidence
The YOLO hybrid model may improve classification confidence by adding explicit material cues
to YOLO’s detection features.
However, it may not greatly improve localization because bounding box prediction still comes
mainly from YOLO.
- Main Risks and Solutions
RiskExplanationSolution
Too many experimentsEleven experiments can be time-
consuming
Implement in phases and
keep YOLO hybrid last
Confusion between
detection and classification
Classical/SVM models do not
detect objects by themselves
Use two evaluation tracks
YOLO FPN extraction may be
difficult
Accessing internal YOLO features
requires extra coding
Treat YOLO hybrid as
advanced/final experiment
Dataset imbalanceSome waste classes may have
fewer examples
Use macro-F1 and class
weights
Feature vector too large2,300 dimensions may be
computationally heavy
Use PCA, autoencoder, and
feature selection
OverfittingSome models may memorize
crop-level features
Use validation set, test set,
and fixed split
## 17. Final Deliverables
By the end of the project, the final submission should include:
DeliverableDescription
Dataset preprocessing codeClass merging, splitting, crop extraction
YOLO training resultsDetection baseline
Feature extraction scriptsClassical, EfficientNet, YOLO FPN
Trained classifiersRF, SVM, autoencoder, attention model
Experiment results tableAll 11 experiments compared
Confusion matricesFor classification experiments
Accuracy vs dimension chartShows compression trade-off
Speed comparisonFeature extraction and classification time
Final discussionBest model and explanation
Final report/presentationAcademic project documentation
## 18. Suggested Final Results Table Format
Your final report should include a table like this:
Exp.MethodFeature
## Dim.
## Accurac
y
## Macro-
## F1
mAP@0.5FPSModel
## Size
E1YOLOv8-nanoNative—————
E2Classical + RF252——N/A——
E3EfficientNet +
## SVM
## 1280——N/A——
E4Raw early fusion2300——N/A——
E5PCA fusion~400–600——N/A——
E6Autoencoder
fusion
## 256——N/A——
E7Feature selection200——N/A——
E8Average votingSeparate——N/A——
E9Weighted votingSeparate——N/A——
E10Attention fusion256——N/A——
E11YOLO hybrid768 + 252—————
Use N/Awhere mAP does not apply.
- Suggested Report Structure for Final Submission
When writing the final academic report, use this structure:
## Abstract 1.
## Introduction 2.
## Problem Statement 3.
## Research Question 4.
## Dataset Description 5.
## Preprocessing 6.
## Feature Extraction Methods 7.
## Fusion Strategies 8.
## Dimensionality Reduction Methods 9.
## Experimental Design 10.
## Evaluation Metrics 11.
## Results 12.
## Discussion 13.
## Limitations 14.
## Conclusion 15.
## References 16.
## 20. Final Feasibility Verdict
This project is viable and strong, as long as it is implemented with the correct separation
between detection and classification.
The most important correction is:
YOLO performs localization and detection. Classical, deep, and fused features mainly
evaluate object crop classification and representation quality.
The project should not claim that Random Forest, SVM, PCA, or attention fusion are full object
detectors unless they are integrated into a full detection pipeline with bounding boxes and
confidence scores.
The strongest final version is:
Train YOLOv8-nano as the detection baseline, then conduct a systematic representation
study on waste object crops using classical features, EfficientNet features, fusion methods,
and compression techniques. Finally, test whether YOLO’s detection features can be
improved through a hybrid fusion model.
## Project Description
Thursday, 14 May 20261:59 PM

Comprehensive Pre-Implementation Report
Compression-Aware Multiscale Feature Fusion for Waste
Object Detection and Classification
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
object detection and classificationacross all object sizes.
## 2. Proposed Project Title
## Recommended Title
Compression-Aware Multiscale Feature Fusion for Waste Object Detection and Classification
Alternative title:
Multiscale Feature Fusion and Dimensionality Reduction for Waste Object Detection
The first title is stronger because it clearly shows the two main technical ideas:
Feature fusion 1.
Dimensionality reduction / compression 2.
## 3. Main Research Question
The main research question is:
How can classical visual features, pretrained deep neural features, and YOLO detection
features be fused and compressed to improve waste-object classification accuracy while
reducing feature dimensionality and computational cost?
This question is clear, realistic, and directly connected to Computer Vision.
## 4. Project Objectives
The project has five main objectives:
ObjectiveExplanation
Compare individual feature
types
Test classical, deep, and YOLO-based features separately
Study feature fusionCompare early fusion, late fusion, attention fusion, and YOLO
hybrid fusion
Apply dimensionality
reduction
Use PCA, autoencoder compression, and feature selection
Evaluate accuracy-efficiency
trade-off
Compare accuracy, F1-score, mAP, speed, feature size, and
model size
Identify the best practical
configuration
Recommend the most accurate and efficient representation for
waste detection/classification
## 5. Dataset
The project uses the Kaggle Trash Detection Image Dataset, which contains 4,142 imageswith
bounding box annotations in YOLO format. The original dataset includes seven classes:
Cardboard, Glass, Metal, Paper, Plastic, Trash, and Biodegradable. In the proposed project,
Biodegradable is merged with Trash, producing a final six-class taxonomy.
## Final Class List
Original ClassFinal Class
CardboardCardboard
GlassGlass
MetalMetal
PaperPaper
PlasticPlastic
TrashTrash
BiodegradableTrash
The final classification problem therefore contains six material categories.
## 6. Dataset Split
The dataset will be divided using stratified sampling:
SplitPercentageApproximate Images
## Training70%~2,900
## Validation15%~620
## Test15%~620
The split must be done at the image level, not object level. This prevents objects from the same
image appearing in both training and test sets.
Stratification is important because some classes, especially Metal, may appear less frequently
than others. The split should preserve class distribution as much as possible.
## 7. Important Methodological Clarification
This project contains two related but different tasks:
## Task A: Object Detection
Object detection means the system outputs:
bounding box + class label + confidence score
This applies mainly to:
YOLOv8-nano baseline 1.
YOLO hybrid model 2.
Detection metrics such as mAP@0.5and mAP@0.5:0.95are valid here.
## Task B: Object Crop Classification
Object crop classification means the object location is already known, and the model only
predicts the material class.
This applies to:
Classical feature classifier 1.
Deep feature classifier 2.
Early fusion models 3.
PCA fusion 4.
Autoencoder fusion 5.
Feature-selection fusion 6.
Late fusion 7.
Attention fusion 8.
These models do not perform full detection by themselves. They classify cropped object regions.
So their main metrics should be:
accuracy, macro-F1, per-class F1, feature dimension, extraction time, and classification
time
This distinction is very important. It makes the methodology technically correct.
## 8. General Pipeline
The full project pipeline will follow these stages:
## Stage 1: Dataset Preparation
Load images and YOLO annotation files. 1.
Convert YOLO labels into bounding box coordinates. 2.
Merge Biodegradable into Trash. 3.
Split dataset into train, validation, and test sets. 4.
Crop object regions from images using ground-truth boxes. 5.
Resize crops when needed, especially for EfficientNet. 6.
Stage 2: YOLO Training
A YOLOv8-nano detector will be trained on the six-class dataset.
YOLO will be used for two purposes:
As a detection baseline. 1.
As a region proposal/localization module for hybrid experiments. 2.
## Stage 3: Feature Extraction
For each object crop, three types of features will be extracted:
Classical hand-crafted features 1.
Deep EfficientNetB0 features 2.
YOLO Feature Pyramid Network features 3.
Stage 4: Fusion and Compression
Different fusion strategies will be tested:
Early feature concatenation 1.
PCA dimensionality reduction 2.
Autoencoder compression 3.
Feature selection 4.
Late probability fusion 5.
Attention-based fusion 6.
YOLO hybrid fusion 7.
## Stage 5: Evaluation
The project will evaluate both detection and classification performance.
Detection models will use mAP-based metrics.
Feature/classification models will use accuracy, macro-F1, per-class F1, time, and feature size.
## 9. Feature Extraction Design
## 9.1 Classical Feature Branch
The classical feature branch extracts interpretable visual properties from object crops.
Expected feature size: 252 dimensions
Feature TypeDimensionsPurpose
Edge features5Capture boundary and gradient information
Shape features19Describe object geometry
LBP texture features119Capture local texture patterns
GLCM texture features48Capture texture regularity and direction
HSV color histogram24Capture color distribution
Spatial pyramid features28Preserve coarse spatial layout
Lab perceptual color features7Represent human-perceived color differences
Material-specific cues2Detect highlight/metallic-like properties
Total252Full classical vector
This branch is useful because waste categories often depend on visible material properties such
as color, texture, reflectiveness, and shape.
## 9.2 Deep Hierarchical Feature Branch
The deep feature branch uses a pretrained EfficientNetB0 model.
The model is used as a feature extractor, not as a fully fine-tuned classifier.
Feature LevelDimensionsMeaning
Early-layer features256Edges, corners, colors, low-level texture
Mid-layer features512Material surfaces, local structures
Late-layer features1,280High-level semantic appearance
Total deep features2,048Full EfficientNet representation
The early, mid, and late features capture different levels of visual information.
9.3 YOLO Detection Feature Branch
YOLOv8-nano produces multiscale feature maps through its Feature Pyramid Network.
YOLO Feature MapDimensionsRole
P3256Higher-resolution features
P4256Medium-scale features
P5256Lower-resolution semantic features
Total YOLO FPN features768Detection-specific representation
These features are useful because they are directly optimized for object detection.
- Experiments to Be Conducted
The project should include eleven experimental configurations.
## Experiment Group 1: Baseline Experiments
These experiments establish the performance of each representation type before fusion.
ExperimentNameFeature SourceModelEvaluation Type
E1YOLOv8-nano
baseline
Full image YOLO
features
YOLO detectorDetection
E2Classical-only baseline252 classical featuresRandom
## Forest
## Classification
E3Deep-only baselineEfficientNet late
features
SVMClassification
E1: YOLOv8-Nano Baseline
This experiment trains YOLOv8-nano on the waste dataset.
## Purpose:
Establish the main object detection baseline.
## Outputs:
Bounding boxes •
Class labels •
Confidence scores •
## Metrics:
mAP@0.5 •
mAP@0.5:0.95 •
Per-class AP •
## FPS •
Model size •
E2: Classical-Only Baseline
This experiment extracts 252-dimensional classical features from each object crop and trains a
Random Forest classifier.
## Purpose:
Test how well interpretable visual features classify waste materials.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
E3: Deep-Only Baseline
This experiment extracts EfficientNetB0 late-layer features and trains an SVM classifier.
## Purpose:
Test how well pretrained deep semantic features classify waste objects.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
## Experiment Group 2: Early Fusion Experiments
Early fusion means features are combined first, then one classifier is trained.
## Experime
nt
NameFeatures UsedReduction MethodClassifier
E4Raw early fusionClassical + EfficientNet
early + mid + late
NoneRandom Forest
E5PCA fusionSame fused vectorPCA, 95% varianceSVM
E6Autoencoder
fusion
Same fused vector256-dimensional
bottleneck
Neural classifier
## / SVM
E7Feature-
selection fusion
Same fused vectorTop 200 featuresRandom Forest
The full early-fusion vector is:
Feature SourceDimensions
Classical features252
EfficientNet early256
EfficientNet mid512
EfficientNet late1,280
## Total2,300
## E4: Raw Early Fusion
The 2,300-dimensional fused feature vector is used directly.
## Purpose:
Test whether combining all available classical and deep features improves classification
performance.
Main question:
Does raw feature concatenation outperform individual feature branches?
E5: PCA Fusion
The 2,300-dimensional vector is compressed using PCA while retaining 95% of variance.
## Purpose:
Test whether linear dimensionality reduction can reduce feature size without large
accuracy loss.
Expected output:
Reduced feature vector, probably around 400–600 dimensions •
Classification results after compression •
## E6: Autoencoder Fusion
A shallow autoencoder compresses the 2,300-dimensional vector into a 256-dimensional
bottleneck.
Suggested structure:
## Input 2300 → Dense 512 → Dense 256 → Dense 512 → Output 2300
The 256-dimensional bottleneck is then used for classification.
## Purpose:
Test whether nonlinear compression preserves information better than PCA.
E7: Feature-Selection Fusion
A Random Forest is used to rank feature importance.
The top 200 features are selected and used for classification.
## Purpose:
Test whether only a small subset of the fused features is enough for strong classification
performance.
This experiment is useful because it may reveal which feature groups are most informative.
## Experiment Group 3: Late Fusion Experiments
Late fusion means separate classifiers are trained first, then their predictions are combined.
ExperimentNameModels CombinedFusion Method
E8Average votingClassical classifier + deep
classifier
Equal probability averaging
E9Weighted votingClassical classifier + deep
classifier
## Validation-optimized
weights
## E8: Average Voting
The classical classifier and deep classifier each produce class probabilities.
The final prediction is:
average of both probability vectors
## Purpose:
Test whether simple decision-level fusion improves performance.
## E9: Weighted Voting
The classical and deep probabilities are combined using optimized weights.
## Example:
Final prediction = 0.4 classical + 0.6 deep
The best weights are selected using the validation set.
## Purpose:
Test whether one representation should be trusted more than another.
This may show that deep features are generally stronger, but classical features still add useful
material-specific information.
Experiment Group 4: Attention-Based Fusion
ExperimentNameFeatures UsedMethod
E10Attention fusionClassical + EfficientNet late featuresLearned gating network
## E10: Attention Fusion
This experiment trains a lightweight neural network that learns how much attention to give to
each feature branch.
General architecture:
Project classical features from 252 dimensions to 256 dimensions. 1.
Project EfficientNet late features from 1,280 dimensions to 256 dimensions. 2.
Concatenate both projected vectors. 3.
Learn attention weights for classical and deep features. 4.
Produce one fused 256-dimensional vector. 5.
Classify the object. 6.
## Purpose:
Test whether the system can dynamically choose which representation is more useful for
each object.
Example interpretation:
Cardboard may rely more on color and texture. •
Glass may rely more on shape, transparency, and deep visual patterns. •
Metal may rely on reflectiveness and deep features. •
This experiment is advanced and gives the project a strong research contribution.
Experiment Group 5: YOLO Hybrid Experiment
## Experimen
t
NameFeatures UsedMethodEvaluation
## E11YOLO
hybrid
YOLO FPN + classical
features
## Fusion/re-scoring
module
## Detection +
classification
E11: YOLO Hybrid
This experiment combines YOLO’s internal detection features with classical hand-crafted
features from the same detected region.
## Purpose:
Test whether YOLO detection features become stronger when combined with explicit
color, texture, and material cues.
Suggested implementation:
Run YOLO on the image. 1.
Extract predicted bounding boxes. 2.
Crop each detected object region. 3.
Extract classical features from the crop. 4.
Extract YOLO FPN features corresponding to the detected region. 5.
Concatenate YOLO FPN features with classical features. 6.
Use a classifier or re-scoring module to refine the class prediction. 7.
Final output:
YOLO bounding box •
Refined class prediction •
Refined confidence score •
Because this system still outputs bounding boxes, class labels, and confidence scores, it can be
evaluated using detection metrics.
## 11. Final Experiment Count
The project contains 11 experiments.
GroupExperimentsCount
BaselinesE1, E2, E33
Early fusionE4, E5, E6, E74
Late fusionE8, E92
Attention fusionE101
YOLO hybridE111
## Total11
So in the report, do not write “thirteen experiments.”
## Use:
eleven experimental configurations
or:
all experimental configurations
## 12. Evaluation Protocol
The evaluation should be divided into two tracks.
## Track A: Detection Evaluation
Used for:
E1: YOLOv8-nano baseline •
E11: YOLO hybrid •
MetricMeaning
mAP@0.5Detection performance at IoU threshold 0.5
mAP@0.5:0.95Stricter detection/localization metric
Per-class APDetection performance for each waste class
PrecisionCorrectness of positive detections
RecallAbility to find existing objects
FPSReal-time speed
Model footprintDeployment size
## Track B: Classification / Representation Evaluation
Used for:
## E2: Classical-only •
## E3: Deep-only •
E4: Raw early fusion •
E5: PCA fusion •
E6: Autoencoder fusion •
E7: Feature-selection fusion •
E8: Average voting •
E9: Weighted voting •
E10: Attention fusion •
MetricMeaning
AccuracyOverall classification correctness
Macro-F1Balanced performance across all classes
Per-class F1Performance for each waste material
Confusion matrixShows which classes are confused
Feature dimensionSize of the representation
Feature extraction timeCost of producing the features
Classification timeCost of predicting the class
Model sizeStorage requirement
Macro-F1 is especially important because the dataset may be imbalanced.
## 13. Controlled Variables
To ensure fair comparison, the following conditions should remain fixed:
Controlled VariableDecision
Dataset splitSame train/validation/test split for all experiments
Class mappingSame six-class taxonomy
Crop generationSame ground-truth crops for classification experiments
Image preprocessingSame resizing and normalization rules
Classifier training dataSame training objects
Test setSame test objects/images
Random seedFixed, for example 42
YOLO confidence thresholdFixed, for example 0.25
NMS IoU thresholdFixed, for example 0.5
## 14. Recommended Implementation Order
To make the project manageable, implement the experiments in this order:
Phase 1: Dataset and YOLO Baseline
Prepare dataset. 1.
Merge classes. 2.
Train YOLOv8-nano. 3.
Evaluate YOLO detection performance. 4.
## Output:
YOLO baseline results.
## Phase 2: Object Crop Dataset
Extract object crops from ground-truth bounding boxes. 1.
Save crop images and labels. 2.
Prepare train/validation/test crop-level datasets. 3.
## Output:
Clean crop classification dataset.
## Phase 3: Baseline Feature Models
Extract classical features. 1.
## Train Random Forest. 2.
Extract EfficientNet features. 3.
Train SVM. 4.
Compare classical-only and deep-only baselines. 5.
## Output:
Baseline representation results.
Phase 4: Early Fusion and Compression
Concatenate classical + EfficientNet features. 1.
Train raw fusion model. 2.
Apply PCA. 3.
Train PCA-fusion classifier. 4.
Train autoencoder. 5.
Extract bottleneck features. 6.
Train autoencoder-fusion classifier. 7.
Perform feature selection. 8.
Train selected-feature classifier. 9.
## Output:
Accuracy vs feature-dimension comparison.
## Phase 5: Late Fusion
Generate probability outputs from classical and deep classifiers. 1.
Apply average voting. 2.
Optimize voting weights on validation set. 3.
Evaluate on test set. 4.
## Output:
Decision-level fusion comparison.
## Phase 6: Attention Fusion
Build attention/gating network. 1.
Train on classical + deep features. 2.
Evaluate classification performance. 3.
Analyze attention weights by class. 4.
## Output:
Interpretable learned fusion model.
Phase 7: YOLO Hybrid
Extract YOLO FPN features. 1.
Combine YOLO FPN features with classical crop features. 2.
Train hybrid classifier/re-scoring module. 3.
Evaluate detection/classification performance. 4.
## Output:
Final hybrid detection result.
This is the most technically challenging experiment, so it should be done after the main feature-
fusion experiments are working.
## 15. Expected Results
The expected findings should be written carefully, not as guaranteed results.
## Expected Finding 1: Fusion Improves Classification
Classical features may perform well on materials with distinctive color and texture, such as
cardboard and paper.
Deep features may perform better on visually complex materials such as glass, metal, and
plastic.
Therefore, fusion is expected to outperform either feature type alone.
Expected Finding 2: PCA May Preserve Most Performance
PCA is expected to reduce the 2,300-dimensional fused vector significantly while preserving
most classification performance.
This would show that the full fused vector contains redundancy.
Expected Finding 3: Autoencoder May Help at Strong
## Compression
At very low dimensions, autoencoder compression may preserve nonlinear feature relationships
better than PCA.
This is useful if the final system is intended for deployment on limited hardware.
## Expected Finding 4: Attention Fusion May Be More
## Interpretable
The attention model may learn different branch weights for different waste classes.
For example:
ClassPossible Dominant Features
CardboardClassical color/texture
PaperClassical texture/color
GlassDeep visual patterns
MetalDeep + reflectiveness cues
PlasticMixed classical/deep
TrashMixed features
Expected Finding 5: YOLO Hybrid May Improve Class
## Confidence
The YOLO hybrid model may improve classification confidence by adding explicit material cues
to YOLO’s detection features.
However, it may not greatly improve localization because bounding box prediction still comes
mainly from YOLO.
- Main Risks and Solutions
RiskExplanationSolution
Too many experimentsEleven experiments can be time-
consuming
Implement in phases and
keep YOLO hybrid last
Confusion between
detection and classification
Classical/SVM models do not
detect objects by themselves
Use two evaluation tracks
YOLO FPN extraction may be
difficult
Accessing internal YOLO features
requires extra coding
Treat YOLO hybrid as
advanced/final experiment
Dataset imbalanceSome waste classes may have
fewer examples
Use macro-F1 and class
weights
Feature vector too large2,300 dimensions may be
computationally heavy
Use PCA, autoencoder, and
feature selection
OverfittingSome models may memorize
crop-level features
Use validation set, test set,
and fixed split
## 17. Final Deliverables
By the end of the project, the final submission should include:
DeliverableDescription
Dataset preprocessing codeClass merging, splitting, crop extraction
YOLO training resultsDetection baseline
Feature extraction scriptsClassical, EfficientNet, YOLO FPN
Trained classifiersRF, SVM, autoencoder, attention model
Experiment results tableAll 11 experiments compared
Confusion matricesFor classification experiments
Accuracy vs dimension chartShows compression trade-off
Speed comparisonFeature extraction and classification time
Final discussionBest model and explanation
Final report/presentationAcademic project documentation
## 18. Suggested Final Results Table Format
Your final report should include a table like this:
Exp.MethodFeature
## Dim.
## Accurac
y
## Macro-
## F1
mAP@0.5FPSModel
## Size
E1YOLOv8-nanoNative—————
E2Classical + RF252——N/A——
E3EfficientNet +
## SVM
## 1280——N/A——
E4Raw early fusion2300——N/A——
E5PCA fusion~400–600——N/A——
E6Autoencoder
fusion
## 256——N/A——
E7Feature selection200——N/A——
E8Average votingSeparate——N/A——
E9Weighted votingSeparate——N/A——
E10Attention fusion256——N/A——
E11YOLO hybrid768 + 252—————
Use N/Awhere mAP does not apply.
- Suggested Report Structure for Final Submission
When writing the final academic report, use this structure:
## Abstract 1.
## Introduction 2.
## Problem Statement 3.
## Research Question 4.
## Dataset Description 5.
## Preprocessing 6.
## Feature Extraction Methods 7.
## Fusion Strategies 8.
## Dimensionality Reduction Methods 9.
## Experimental Design 10.
## Evaluation Metrics 11.
## Results 12.
## Discussion 13.
## Limitations 14.
## Conclusion 15.
## References 16.
## 20. Final Feasibility Verdict
This project is viable and strong, as long as it is implemented with the correct separation
between detection and classification.
The most important correction is:
YOLO performs localization and detection. Classical, deep, and fused features mainly
evaluate object crop classification and representation quality.
The project should not claim that Random Forest, SVM, PCA, or attention fusion are full object
detectors unless they are integrated into a full detection pipeline with bounding boxes and
confidence scores.
The strongest final version is:
Train YOLOv8-nano as the detection baseline, then conduct a systematic representation
study on waste object crops using classical features, EfficientNet features, fusion methods,
and compression techniques. Finally, test whether YOLO’s detection features can be
improved through a hybrid fusion model.
## Project Description
Thursday, 14 May 20261:59 PM

Comprehensive Pre-Implementation Report
Compression-Aware Multiscale Feature Fusion for Waste
Object Detection and Classification
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
object detection and classificationacross all object sizes.
## 2. Proposed Project Title
## Recommended Title
Compression-Aware Multiscale Feature Fusion for Waste Object Detection and Classification
Alternative title:
Multiscale Feature Fusion and Dimensionality Reduction for Waste Object Detection
The first title is stronger because it clearly shows the two main technical ideas:
Feature fusion 1.
Dimensionality reduction / compression 2.
## 3. Main Research Question
The main research question is:
How can classical visual features, pretrained deep neural features, and YOLO detection
features be fused and compressed to improve waste-object classification accuracy while
reducing feature dimensionality and computational cost?
This question is clear, realistic, and directly connected to Computer Vision.
## 4. Project Objectives
The project has five main objectives:
ObjectiveExplanation
Compare individual feature
types
Test classical, deep, and YOLO-based features separately
Study feature fusionCompare early fusion, late fusion, attention fusion, and YOLO
hybrid fusion
Apply dimensionality
reduction
Use PCA, autoencoder compression, and feature selection
Evaluate accuracy-efficiency
trade-off
Compare accuracy, F1-score, mAP, speed, feature size, and
model size
Identify the best practical
configuration
Recommend the most accurate and efficient representation for
waste detection/classification
## 5. Dataset
The project uses the Kaggle Trash Detection Image Dataset, which contains 4,142 imageswith
bounding box annotations in YOLO format. The original dataset includes seven classes:
Cardboard, Glass, Metal, Paper, Plastic, Trash, and Biodegradable. In the proposed project,
Biodegradable is merged with Trash, producing a final six-class taxonomy.
## Final Class List
Original ClassFinal Class
CardboardCardboard
GlassGlass
MetalMetal
PaperPaper
PlasticPlastic
TrashTrash
BiodegradableTrash
The final classification problem therefore contains six material categories.
## 6. Dataset Split
The dataset will be divided using stratified sampling:
SplitPercentageApproximate Images
## Training70%~2,900
## Validation15%~620
## Test15%~620
The split must be done at the image level, not object level. This prevents objects from the same
image appearing in both training and test sets.
Stratification is important because some classes, especially Metal, may appear less frequently
than others. The split should preserve class distribution as much as possible.
## 7. Important Methodological Clarification
This project contains two related but different tasks:
## Task A: Object Detection
Object detection means the system outputs:
bounding box + class label + confidence score
This applies mainly to:
YOLOv8-nano baseline 1.
YOLO hybrid model 2.
Detection metrics such as mAP@0.5and mAP@0.5:0.95are valid here.
## Task B: Object Crop Classification
Object crop classification means the object location is already known, and the model only
predicts the material class.
This applies to:
Classical feature classifier 1.
Deep feature classifier 2.
Early fusion models 3.
PCA fusion 4.
Autoencoder fusion 5.
Feature-selection fusion 6.
Late fusion 7.
Attention fusion 8.
These models do not perform full detection by themselves. They classify cropped object regions.
So their main metrics should be:
accuracy, macro-F1, per-class F1, feature dimension, extraction time, and classification
time
This distinction is very important. It makes the methodology technically correct.
## 8. General Pipeline
The full project pipeline will follow these stages:
## Stage 1: Dataset Preparation
Load images and YOLO annotation files. 1.
Convert YOLO labels into bounding box coordinates. 2.
Merge Biodegradable into Trash. 3.
Split dataset into train, validation, and test sets. 4.
Crop object regions from images using ground-truth boxes. 5.
Resize crops when needed, especially for EfficientNet. 6.
Stage 2: YOLO Training
A YOLOv8-nano detector will be trained on the six-class dataset.
YOLO will be used for two purposes:
As a detection baseline. 1.
As a region proposal/localization module for hybrid experiments. 2.
## Stage 3: Feature Extraction
For each object crop, three types of features will be extracted:
Classical hand-crafted features 1.
Deep EfficientNetB0 features 2.
YOLO Feature Pyramid Network features 3.
Stage 4: Fusion and Compression
Different fusion strategies will be tested:
Early feature concatenation 1.
PCA dimensionality reduction 2.
Autoencoder compression 3.
Feature selection 4.
Late probability fusion 5.
Attention-based fusion 6.
YOLO hybrid fusion 7.
## Stage 5: Evaluation
The project will evaluate both detection and classification performance.
Detection models will use mAP-based metrics.
Feature/classification models will use accuracy, macro-F1, per-class F1, time, and feature size.
## 9. Feature Extraction Design
## 9.1 Classical Feature Branch
The classical feature branch extracts interpretable visual properties from object crops.
Expected feature size: 252 dimensions
Feature TypeDimensionsPurpose
Edge features5Capture boundary and gradient information
Shape features19Describe object geometry
LBP texture features119Capture local texture patterns
GLCM texture features48Capture texture regularity and direction
HSV color histogram24Capture color distribution
Spatial pyramid features28Preserve coarse spatial layout
Lab perceptual color features7Represent human-perceived color differences
Material-specific cues2Detect highlight/metallic-like properties
Total252Full classical vector
This branch is useful because waste categories often depend on visible material properties such
as color, texture, reflectiveness, and shape.
## 9.2 Deep Hierarchical Feature Branch
The deep feature branch uses a pretrained EfficientNetB0 model.
The model is used as a feature extractor, not as a fully fine-tuned classifier.
Feature LevelDimensionsMeaning
Early-layer features256Edges, corners, colors, low-level texture
Mid-layer features512Material surfaces, local structures
Late-layer features1,280High-level semantic appearance
Total deep features2,048Full EfficientNet representation
The early, mid, and late features capture different levels of visual information.
9.3 YOLO Detection Feature Branch
YOLOv8-nano produces multiscale feature maps through its Feature Pyramid Network.
YOLO Feature MapDimensionsRole
P3256Higher-resolution features
P4256Medium-scale features
P5256Lower-resolution semantic features
Total YOLO FPN features768Detection-specific representation
These features are useful because they are directly optimized for object detection.
- Experiments to Be Conducted
The project should include eleven experimental configurations.
## Experiment Group 1: Baseline Experiments
These experiments establish the performance of each representation type before fusion.
## Experimen
t
NameFeature SourceModelEvaluation Type
E1YOLOv8-nano
baseline
Full image YOLO
features
YOLO detectorDetection
E2Classical-only baseline252 classical featuresRandom
## Forest
## Classification
E3Deep-only baselineEfficientNet late
features
SVMClassification
E1: YOLOv8-Nano Baseline
This experiment trains YOLOv8-nano on the waste dataset.
## Purpose:
Establish the main object detection baseline.
## Outputs:
Bounding boxes •
Class labels •
Confidence scores •
## Metrics:
mAP@0.5 •
mAP@0.5:0.95 •
Per-class AP •
## FPS •
Model size •
E2: Classical-Only Baseline
This experiment extracts 252-dimensional classical features from each object crop and trains a
Random Forest classifier.
## Purpose:
Test how well interpretable visual features classify waste materials.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
E3: Deep-Only Baseline
This experiment extracts EfficientNetB0 late-layer features and trains an SVM classifier.
## Purpose:
Test how well pretrained deep semantic features classify waste objects.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
## Experiment Group 2: Early Fusion Experiments
Early fusion means features are combined first, then one classifier is trained.
## Experime
nt
NameFeatures UsedReduction MethodClassifier
E4Raw early fusionClassical + EfficientNet
early + mid + late
NoneRandom Forest
E5PCA fusionSame fused vectorPCA, 95% varianceSVM
E6Autoencoder
fusion
Same fused vector256-dimensional
bottleneck
Neural classifier
## / SVM
E7Feature-
selection fusion
Same fused vectorTop 200 featuresRandom Forest
The full early-fusion vector is:
Feature SourceDimensions
Classical features252
EfficientNet early256
EfficientNet mid512
EfficientNet late1,280
## Total2,300
## E4: Raw Early Fusion
The 2,300-dimensional fused feature vector is used directly.
## Purpose:
Test whether combining all available classical and deep features improves classification
performance.
Main question:
Does raw feature concatenation outperform individual feature branches?
E5: PCA Fusion
The 2,300-dimensional vector is compressed using PCA while retaining 95% of variance.
## Purpose:
Test whether linear dimensionality reduction can reduce feature size without large
accuracy loss.
Expected output:
Reduced feature vector, probably around 400–600 dimensions •
Classification results after compression •
## E6: Autoencoder Fusion
A shallow autoencoder compresses the 2,300-dimensional vector into a 256-dimensional
bottleneck.
Suggested structure:
## Input 2300 → Dense 512 → Dense 256 → Dense 512 → Output 2300
The 256-dimensional bottleneck is then used for classification.
## Purpose:
Test whether nonlinear compression preserves information better than PCA.
E7: Feature-Selection Fusion
A Random Forest is used to rank feature importance.
The top 200 features are selected and used for classification.
## Purpose:
Test whether only a small subset of the fused features is enough for strong classification
performance.
This experiment is useful because it may reveal which feature groups are most informative.
## Experiment Group 3: Late Fusion Experiments
Late fusion means separate classifiers are trained first, then their predictions are combined.
ExperimentNameModels CombinedFusion Method
E8Average votingClassical classifier + deep
classifier
Equal probability averaging
E9Weighted votingClassical classifier + deep
classifier
## Validation-optimized
weights
## E8: Average Voting
The classical classifier and deep classifier each produce class probabilities.
The final prediction is:
average of both probability vectors
## Purpose:
Test whether simple decision-level fusion improves performance.
## E9: Weighted Voting
The classical and deep probabilities are combined using optimized weights.
## Example:
Final prediction = 0.4 classical + 0.6 deep
The best weights are selected using the validation set.
## Purpose:
Test whether one representation should be trusted more than another.
This may show that deep features are generally stronger, but classical features still add useful
material-specific information.
Experiment Group 4: Attention-Based Fusion
ExperimentNameFeatures UsedMethod
E10Attention fusionClassical + EfficientNet late featuresLearned gating network
## E10: Attention Fusion
This experiment trains a lightweight neural network that learns how much attention to give to
each feature branch.
General architecture:
Project classical features from 252 dimensions to 256 dimensions. 1.
Project EfficientNet late features from 1,280 dimensions to 256 dimensions. 2.
Concatenate both projected vectors. 3.
Learn attention weights for classical and deep features. 4.
Produce one fused 256-dimensional vector. 5.
Classify the object. 6.
## Purpose:
Test whether the system can dynamically choose which representation is more useful for
each object.
Example interpretation:
Cardboard may rely more on color and texture. •
Glass may rely more on shape, transparency, and deep visual patterns. •
Metal may rely on reflectiveness and deep features. •
This experiment is advanced and gives the project a strong research contribution.
Experiment Group 5: YOLO Hybrid Experiment
## Experimen
t
NameFeatures UsedMethodEvaluation
## E11YOLO
hybrid
YOLO FPN + classical
features
## Fusion/re-scoring
module
## Detection +
classification
E11: YOLO Hybrid
This experiment combines YOLO’s internal detection features with classical hand-crafted
features from the same detected region.
## Purpose:
Test whether YOLO detection features become stronger when combined with explicit
color, texture, and material cues.
Suggested implementation:
Run YOLO on the image. 1.
Extract predicted bounding boxes. 2.
Crop each detected object region. 3.
Extract classical features from the crop. 4.
Extract YOLO FPN features corresponding to the detected region. 5.
Concatenate YOLO FPN features with classical features. 6.
Use a classifier or re-scoring module to refine the class prediction. 7.
Final output:
YOLO bounding box •
Refined class prediction •
Refined confidence score •
Because this system still outputs bounding boxes, class labels, and confidence scores, it can be
evaluated using detection metrics.
## 11. Final Experiment Count
The project contains 11 experiments.
GroupExperimentsCount
BaselinesE1, E2, E33
Early fusionE4, E5, E6, E74
Late fusionE8, E92
Attention fusionE101
YOLO hybridE111
## Total11
So in the report, do not write “thirteen experiments.”
## Use:
eleven experimental configurations
or:
all experimental configurations
## 12. Evaluation Protocol
The evaluation should be divided into two tracks.
## Track A: Detection Evaluation
Used for:
E1: YOLOv8-nano baseline •
E11: YOLO hybrid •
MetricMeaning
mAP@0.5Detection performance at IoU threshold 0.5
mAP@0.5:0.95Stricter detection/localization metric
Per-class APDetection performance for each waste class
PrecisionCorrectness of positive detections
RecallAbility to find existing objects
FPSReal-time speed
Model footprintDeployment size
## Track B: Classification / Representation Evaluation
Used for:
## E2: Classical-only •
## E3: Deep-only •
E4: Raw early fusion •
E5: PCA fusion •
E6: Autoencoder fusion •
E7: Feature-selection fusion •
E8: Average voting •
E9: Weighted voting •
E10: Attention fusion •
MetricMeaning
AccuracyOverall classification correctness
Macro-F1Balanced performance across all classes
Per-class F1Performance for each waste material
Confusion matrixShows which classes are confused
Feature dimensionSize of the representation
Feature extraction timeCost of producing the features
Classification timeCost of predicting the class
Model sizeStorage requirement
Macro-F1 is especially important because the dataset may be imbalanced.
## 13. Controlled Variables
To ensure fair comparison, the following conditions should remain fixed:
Controlled VariableDecision
Dataset splitSame train/validation/test split for all experiments
Class mappingSame six-class taxonomy
Crop generationSame ground-truth crops for classification experiments
Image preprocessingSame resizing and normalization rules
Classifier training dataSame training objects
Test setSame test objects/images
Random seedFixed, for example 42
YOLO confidence thresholdFixed, for example 0.25
NMS IoU thresholdFixed, for example 0.5
## 14. Recommended Implementation Order
To make the project manageable, implement the experiments in this order:
Phase 1: Dataset and YOLO Baseline
Prepare dataset. 1.
Merge classes. 2.
Train YOLOv8-nano. 3.
Evaluate YOLO detection performance. 4.
## Output:
YOLO baseline results.
## Phase 2: Object Crop Dataset
Extract object crops from ground-truth bounding boxes. 1.
Save crop images and labels. 2.
Prepare train/validation/test crop-level datasets. 3.
## Output:
Clean crop classification dataset.
## Phase 3: Baseline Feature Models
Extract classical features. 1.
## Train Random Forest. 2.
Extract EfficientNet features. 3.
Train SVM. 4.
Compare classical-only and deep-only baselines. 5.
## Output:
Baseline representation results.
Phase 4: Early Fusion and Compression
Concatenate classical + EfficientNet features. 1.
Train raw fusion model. 2.
Apply PCA. 3.
Train PCA-fusion classifier. 4.
Train autoencoder. 5.
Extract bottleneck features. 6.
Train autoencoder-fusion classifier. 7.
Perform feature selection. 8.
Train selected-feature classifier. 9.
## Output:
Accuracy vs feature-dimension comparison.
## Phase 5: Late Fusion
Generate probability outputs from classical and deep classifiers. 1.
Apply average voting. 2.
Optimize voting weights on validation set. 3.
Evaluate on test set. 4.
## Output:
Decision-level fusion comparison.
## Phase 6: Attention Fusion
Build attention/gating network. 1.
Train on classical + deep features. 2.
Evaluate classification performance. 3.
Analyze attention weights by class. 4.
## Output:
Interpretable learned fusion model.
Phase 7: YOLO Hybrid
Extract YOLO FPN features. 1.
Combine YOLO FPN features with classical crop features. 2.
Train hybrid classifier/re-scoring module. 3.
Evaluate detection/classification performance. 4.
## Output:
Final hybrid detection result.
This is the most technically challenging experiment, so it should be done after the main feature-
fusion experiments are working.
## 15. Expected Results
The expected findings should be written carefully, not as guaranteed results.
## Expected Finding 1: Fusion Improves Classification
Classical features may perform well on materials with distinctive color and texture, such as
cardboard and paper.
Deep features may perform better on visually complex materials such as glass, metal, and
plastic.
Therefore, fusion is expected to outperform either feature type alone.
Expected Finding 2: PCA May Preserve Most Performance
PCA is expected to reduce the 2,300-dimensional fused vector significantly while preserving
most classification performance.
This would show that the full fused vector contains redundancy.
Expected Finding 3: Autoencoder May Help at Strong
## Compression
At very low dimensions, autoencoder compression may preserve nonlinear feature relationships
better than PCA.
This is useful if the final system is intended for deployment on limited hardware.
## Expected Finding 4: Attention Fusion May Be More
## Interpretable
The attention model may learn different branch weights for different waste classes.
For example:
ClassPossible Dominant Features
CardboardClassical color/texture
PaperClassical texture/color
GlassDeep visual patterns
MetalDeep + reflectiveness cues
PlasticMixed classical/deep
TrashMixed features
Expected Finding 5: YOLO Hybrid May Improve Class
## Confidence
The YOLO hybrid model may improve classification confidence by adding explicit material cues
to YOLO’s detection features.
However, it may not greatly improve localization because bounding box prediction still comes
mainly from YOLO.
- Main Risks and Solutions
RiskExplanationSolution
Too many experimentsEleven experiments can be time-
consuming
Implement in phases and
keep YOLO hybrid last
Confusion between
detection and classification
Classical/SVM models do not
detect objects by themselves
Use two evaluation tracks
YOLO FPN extraction may be
difficult
Accessing internal YOLO features
requires extra coding
Treat YOLO hybrid as
advanced/final experiment
Dataset imbalanceSome waste classes may have
fewer examples
Use macro-F1 and class
weights
Feature vector too large2,300 dimensions may be
computationally heavy
Use PCA, autoencoder, and
feature selection
OverfittingSome models may memorize
crop-level features
Use validation set, test set,
and fixed split
## 17. Final Deliverables
By the end of the project, the final submission should include:
DeliverableDescription
Dataset preprocessing codeClass merging, splitting, crop extraction
YOLO training resultsDetection baseline
Feature extraction scriptsClassical, EfficientNet, YOLO FPN
Trained classifiersRF, SVM, autoencoder, attention model
Experiment results tableAll 11 experiments compared
Confusion matricesFor classification experiments
Accuracy vs dimension chartShows compression trade-off
Speed comparisonFeature extraction and classification time
Final discussionBest model and explanation
Final report/presentationAcademic project documentation
## 18. Suggested Final Results Table Format
Your final report should include a table like this:
Exp.MethodFeature
## Dim.
## Accurac
y
## Macro-
## F1
mAP@0.5FPSModel
## Size
E1YOLOv8-nanoNative—————
E2Classical + RF252——N/A——
E3EfficientNet +
## SVM
## 1280——N/A——
E4Raw early fusion2300——N/A——
E5PCA fusion~400–600——N/A——
E6Autoencoder
fusion
## 256——N/A——
E7Feature selection200——N/A——
E8Average votingSeparate——N/A——
E9Weighted votingSeparate——N/A——
E10Attention fusion256——N/A——
E11YOLO hybrid768 + 252—————
Use N/Awhere mAP does not apply.
- Suggested Report Structure for Final Submission
When writing the final academic report, use this structure:
## Abstract 1.
## Introduction 2.
## Problem Statement 3.
## Research Question 4.
## Dataset Description 5.
## Preprocessing 6.
## Feature Extraction Methods 7.
## Fusion Strategies 8.
## Dimensionality Reduction Methods 9.
## Experimental Design 10.
## Evaluation Metrics 11.
## Results 12.
## Discussion 13.
## Limitations 14.
## Conclusion 15.
## References 16.
## 20. Final Feasibility Verdict
This project is viable and strong, as long as it is implemented with the correct separation
between detection and classification.
The most important correction is:
YOLO performs localization and detection. Classical, deep, and fused features mainly
evaluate object crop classification and representation quality.
The project should not claim that Random Forest, SVM, PCA, or attention fusion are full object
detectors unless they are integrated into a full detection pipeline with bounding boxes and
confidence scores.
The strongest final version is:
Train YOLOv8-nano as the detection baseline, then conduct a systematic representation
study on waste object crops using classical features, EfficientNet features, fusion methods,
and compression techniques. Finally, test whether YOLO’s detection features can be
improved through a hybrid fusion model.
## Project Description
Thursday, 14 May 20261:59 PM

Comprehensive Pre-Implementation Report
Compression-Aware Multiscale Feature Fusion for Waste
Object Detection and Classification
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
object detection and classificationacross all object sizes.
## 2. Proposed Project Title
## Recommended Title
Compression-Aware Multiscale Feature Fusion for Waste Object Detection and Classification
Alternative title:
Multiscale Feature Fusion and Dimensionality Reduction for Waste Object Detection
The first title is stronger because it clearly shows the two main technical ideas:
Feature fusion 1.
Dimensionality reduction / compression 2.
## 3. Main Research Question
The main research question is:
How can classical visual features, pretrained deep neural features, and YOLO detection
features be fused and compressed to improve waste-object classification accuracy while
reducing feature dimensionality and computational cost?
This question is clear, realistic, and directly connected to Computer Vision.
## 4. Project Objectives
The project has five main objectives:
ObjectiveExplanation
Compare individual feature
types
Test classical, deep, and YOLO-based features separately
Study feature fusionCompare early fusion, late fusion, attention fusion, and YOLO
hybrid fusion
Apply dimensionality
reduction
Use PCA, autoencoder compression, and feature selection
Evaluate accuracy-efficiency
trade-off
Compare accuracy, F1-score, mAP, speed, feature size, and
model size
Identify the best practical
configuration
Recommend the most accurate and efficient representation for
waste detection/classification
## 5. Dataset
The project uses the Kaggle Trash Detection Image Dataset, which contains 4,142 imageswith
bounding box annotations in YOLO format. The original dataset includes seven classes:
Cardboard, Glass, Metal, Paper, Plastic, Trash, and Biodegradable. In the proposed project,
Biodegradable is merged with Trash, producing a final six-class taxonomy.
## Final Class List
Original ClassFinal Class
CardboardCardboard
GlassGlass
MetalMetal
PaperPaper
PlasticPlastic
TrashTrash
BiodegradableTrash
The final classification problem therefore contains six material categories.
## 6. Dataset Split
The dataset will be divided using stratified sampling:
SplitPercentageApproximate Images
## Training70%~2,900
## Validation15%~620
## Test15%~620
The split must be done at the image level, not object level. This prevents objects from the same
image appearing in both training and test sets.
Stratification is important because some classes, especially Metal, may appear less frequently
than others. The split should preserve class distribution as much as possible.
## 7. Important Methodological Clarification
This project contains two related but different tasks:
## Task A: Object Detection
Object detection means the system outputs:
bounding box + class label + confidence score
This applies mainly to:
YOLOv8-nano baseline 1.
YOLO hybrid model 2.
Detection metrics such as mAP@0.5and mAP@0.5:0.95are valid here.
## Task B: Object Crop Classification
Object crop classification means the object location is already known, and the model only
predicts the material class.
This applies to:
Classical feature classifier 1.
Deep feature classifier 2.
Early fusion models 3.
PCA fusion 4.
Autoencoder fusion 5.
Feature-selection fusion 6.
Late fusion 7.
Attention fusion 8.
These models do not perform full detection by themselves. They classify cropped object regions.
So their main metrics should be:
accuracy, macro-F1, per-class F1, feature dimension, extraction time, and classification
time
This distinction is very important. It makes the methodology technically correct.
## 8. General Pipeline
The full project pipeline will follow these stages:
## Stage 1: Dataset Preparation
Load images and YOLO annotation files. 1.
Convert YOLO labels into bounding box coordinates. 2.
Merge Biodegradable into Trash. 3.
Split dataset into train, validation, and test sets. 4.
Crop object regions from images using ground-truth boxes. 5.
Resize crops when needed, especially for EfficientNet. 6.
Stage 2: YOLO Training
A YOLOv8-nano detector will be trained on the six-class dataset.
YOLO will be used for two purposes:
As a detection baseline. 1.
As a region proposal/localization module for hybrid experiments. 2.
## Stage 3: Feature Extraction
For each object crop, three types of features will be extracted:
Classical hand-crafted features 1.
Deep EfficientNetB0 features 2.
YOLO Feature Pyramid Network features 3.
Stage 4: Fusion and Compression
Different fusion strategies will be tested:
Early feature concatenation 1.
PCA dimensionality reduction 2.
Autoencoder compression 3.
Feature selection 4.
Late probability fusion 5.
Attention-based fusion 6.
YOLO hybrid fusion 7.
## Stage 5: Evaluation
The project will evaluate both detection and classification performance.
Detection models will use mAP-based metrics.
Feature/classification models will use accuracy, macro-F1, per-class F1, time, and feature size.
## 9. Feature Extraction Design
## 9.1 Classical Feature Branch
The classical feature branch extracts interpretable visual properties from object crops.
Expected feature size: 252 dimensions
Feature TypeDimensionsPurpose
Edge features5Capture boundary and gradient information
Shape features19Describe object geometry
LBP texture features119Capture local texture patterns
GLCM texture features48Capture texture regularity and direction
HSV color histogram24Capture color distribution
Spatial pyramid features28Preserve coarse spatial layout
Lab perceptual color features7Represent human-perceived color differences
Material-specific cues2Detect highlight/metallic-like properties
Total252Full classical vector
This branch is useful because waste categories often depend on visible material properties such
as color, texture, reflectiveness, and shape.
## 9.2 Deep Hierarchical Feature Branch
The deep feature branch uses a pretrained EfficientNetB0 model.
The model is used as a feature extractor, not as a fully fine-tuned classifier.
Feature LevelDimensionsMeaning
Early-layer features256Edges, corners, colors, low-level texture
Mid-layer features512Material surfaces, local structures
Late-layer features1,280High-level semantic appearance
Total deep features2,048Full EfficientNet representation
The early, mid, and late features capture different levels of visual information.
9.3 YOLO Detection Feature Branch
YOLOv8-nano produces multiscale feature maps through its Feature Pyramid Network.
YOLO Feature MapDimensionsRole
P3256Higher-resolution features
P4256Medium-scale features
P5256Lower-resolution semantic features
Total YOLO FPN features768Detection-specific representation
These features are useful because they are directly optimized for object detection.
- Experiments to Be Conducted
The project should include eleven experimental configurations.
## Experiment Group 1: Baseline Experiments
These experiments establish the performance of each representation type before fusion.
ExperimentNameFeature SourceModelEvaluation Type
E1YOLOv8-nano
baseline
Full image YOLO
features
YOLO detectorDetection
E2Classical-only baseline252 classical featuresRandom
## Forest
## Classification
E3Deep-only baselineEfficientNet late
features
SVMClassification
E1: YOLOv8-Nano Baseline
This experiment trains YOLOv8-nano on the waste dataset.
## Purpose:
Establish the main object detection baseline.
## Outputs:
Bounding boxes •
Class labels •
Confidence scores •
## Metrics:
mAP@0.5 •
mAP@0.5:0.95 •
Per-class AP •
## FPS •
Model size •
E2: Classical-Only Baseline
This experiment extracts 252-dimensional classical features from each object crop and trains a
Random Forest classifier.
## Purpose:
Test how well interpretable visual features classify waste materials.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
E3: Deep-Only Baseline
This experiment extracts EfficientNetB0 late-layer features and trains an SVM classifier.
## Purpose:
Test how well pretrained deep semantic features classify waste objects.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
## Experiment Group 2: Early Fusion Experiments
Early fusion means features are combined first, then one classifier is trained.
## Experime
nt
NameFeatures UsedReduction MethodClassifier
E4Raw early fusionClassical + EfficientNet
early + mid + late
NoneRandom Forest
E5PCA fusionSame fused vectorPCA, 95% varianceSVM
E6Autoencoder
fusion
Same fused vector256-dimensional
bottleneck
Neural classifier
## / SVM
E7Feature-
selection fusion
Same fused vectorTop 200 featuresRandom Forest
The full early-fusion vector is:
Feature SourceDimensions
Classical features252
EfficientNet early256
EfficientNet mid512
EfficientNet late1,280
## Total2,300
## E4: Raw Early Fusion
The 2,300-dimensional fused feature vector is used directly.
## Purpose:
Test whether combining all available classical and deep features improves classification
performance.
Main question:
Does raw feature concatenation outperform individual feature branches?
E5: PCA Fusion
The 2,300-dimensional vector is compressed using PCA while retaining 95% of variance.
## Purpose:
Test whether linear dimensionality reduction can reduce feature size without large
accuracy loss.
Expected output:
Reduced feature vector, probably around 400–600 dimensions •
Classification results after compression •
## E6: Autoencoder Fusion
A shallow autoencoder compresses the 2,300-dimensional vector into a 256-dimensional
bottleneck.
Suggested structure:
## Input 2300 → Dense 512 → Dense 256 → Dense 512 → Output 2300
The 256-dimensional bottleneck is then used for classification.
## Purpose:
Test whether nonlinear compression preserves information better than PCA.
E7: Feature-Selection Fusion
A Random Forest is used to rank feature importance.
The top 200 features are selected and used for classification.
## Purpose:
Test whether only a small subset of the fused features is enough for strong classification
performance.
This experiment is useful because it may reveal which feature groups are most informative.
## Experiment Group 3: Late Fusion Experiments
Late fusion means separate classifiers are trained first, then their predictions are combined.
ExperimentNameModels CombinedFusion Method
E8Average votingClassical classifier + deep
classifier
Equal probability averaging
E9Weighted
voting
Classical classifier + deep
classifier
## Validation-optimized
weights
## E8: Average Voting
The classical classifier and deep classifier each produce class probabilities.
The final prediction is:
average of both probability vectors
## Purpose:
Test whether simple decision-level fusion improves performance.
## E9: Weighted Voting
The classical and deep probabilities are combined using optimized weights.
## Example:
Final prediction = 0.4 classical + 0.6 deep
The best weights are selected using the validation set.
## Purpose:
Test whether one representation should be trusted more than another.
This may show that deep features are generally stronger, but classical features still add useful
material-specific information.
Experiment Group 4: Attention-Based Fusion
ExperimentNameFeatures UsedMethod
E10Attention fusionClassical + EfficientNet late featuresLearned gating network
## E10: Attention Fusion
This experiment trains a lightweight neural network that learns how much attention to give to
each feature branch.
General architecture:
Project classical features from 252 dimensions to 256 dimensions. 1.
Project EfficientNet late features from 1,280 dimensions to 256 dimensions. 2.
Concatenate both projected vectors. 3.
Learn attention weights for classical and deep features. 4.
Produce one fused 256-dimensional vector. 5.
Classify the object. 6.
## Purpose:
Test whether the system can dynamically choose which representation is more useful for
each object.
Example interpretation:
Cardboard may rely more on color and texture. •
Glass may rely more on shape, transparency, and deep visual patterns. •
Metal may rely on reflectiveness and deep features. •
This experiment is advanced and gives the project a strong research contribution.
Experiment Group 5: YOLO Hybrid Experiment
## Experimen
t
NameFeatures UsedMethodEvaluation
## E11YOLO
hybrid
YOLO FPN + classical
features
## Fusion/re-scoring
module
## Detection +
classification
E11: YOLO Hybrid
This experiment combines YOLO’s internal detection features with classical hand-crafted
features from the same detected region.
## Purpose:
Test whether YOLO detection features become stronger when combined with explicit
color, texture, and material cues.
Suggested implementation:
Run YOLO on the image. 1.
Extract predicted bounding boxes. 2.
Crop each detected object region. 3.
Extract classical features from the crop. 4.
Extract YOLO FPN features corresponding to the detected region. 5.
Concatenate YOLO FPN features with classical features. 6.
Use a classifier or re-scoring module to refine the class prediction. 7.
Final output:
YOLO bounding box •
Refined class prediction •
Refined confidence score •
Because this system still outputs bounding boxes, class labels, and confidence scores, it can be
evaluated using detection metrics.
## 11. Final Experiment Count
The project contains 11 experiments.
GroupExperimentsCount
BaselinesE1, E2, E33
Early fusionE4, E5, E6, E74
Late fusionE8, E92
Attention fusionE101
YOLO hybridE111
## Total11
So in the report, do not write “thirteen experiments.”
## Use:
eleven experimental configurations
or:
all experimental configurations
## 12. Evaluation Protocol
The evaluation should be divided into two tracks.
## Track A: Detection Evaluation
Used for:
E1: YOLOv8-nano baseline •
E11: YOLO hybrid •
MetricMeaning
mAP@0.5Detection performance at IoU threshold 0.5
mAP@0.5:0.95Stricter detection/localization metric
Per-class APDetection performance for each waste class
PrecisionCorrectness of positive detections
RecallAbility to find existing objects
FPSReal-time speed
Model footprintDeployment size
## Track B: Classification / Representation Evaluation
Used for:
## E2: Classical-only •
## E3: Deep-only •
E4: Raw early fusion •
E5: PCA fusion •
E6: Autoencoder fusion •
E7: Feature-selection fusion •
E8: Average voting •
E9: Weighted voting •
E10: Attention fusion •
MetricMeaning
AccuracyOverall classification correctness
Macro-F1Balanced performance across all classes
Per-class F1Performance for each waste material
Confusion matrixShows which classes are confused
Feature dimensionSize of the representation
Feature extraction timeCost of producing the features
Classification timeCost of predicting the class
Model sizeStorage requirement
Macro-F1 is especially important because the dataset may be imbalanced.
## 13. Controlled Variables
To ensure fair comparison, the following conditions should remain fixed:
Controlled VariableDecision
Dataset splitSame train/validation/test split for all experiments
Class mappingSame six-class taxonomy
Crop generationSame ground-truth crops for classification experiments
Image preprocessingSame resizing and normalization rules
Classifier training dataSame training objects
Test setSame test objects/images
Random seedFixed, for example 42
YOLO confidence thresholdFixed, for example 0.25
NMS IoU thresholdFixed, for example 0.5
## 14. Recommended Implementation Order
To make the project manageable, implement the experiments in this order:
Phase 1: Dataset and YOLO Baseline
Prepare dataset. 1.
Merge classes. 2.
Train YOLOv8-nano. 3.
Evaluate YOLO detection performance. 4.
## Output:
YOLO baseline results.
## Phase 2: Object Crop Dataset
Extract object crops from ground-truth bounding boxes. 1.
Save crop images and labels. 2.
Prepare train/validation/test crop-level datasets. 3.
## Output:
Clean crop classification dataset.
## Phase 3: Baseline Feature Models
Extract classical features. 1.
## Train Random Forest. 2.
Extract EfficientNet features. 3.
Train SVM. 4.
Compare classical-only and deep-only baselines. 5.
## Output:
Baseline representation results.
Phase 4: Early Fusion and Compression
Concatenate classical + EfficientNet features. 1.
Train raw fusion model. 2.
Apply PCA. 3.
Train PCA-fusion classifier. 4.
Train autoencoder. 5.
Extract bottleneck features. 6.
Train autoencoder-fusion classifier. 7.
Perform feature selection. 8.
Train selected-feature classifier. 9.
## Output:
Accuracy vs feature-dimension comparison.
## Phase 5: Late Fusion
Generate probability outputs from classical and deep classifiers. 1.
Apply average voting. 2.
Optimize voting weights on validation set. 3.
Evaluate on test set. 4.
## Output:
Decision-level fusion comparison.
## Phase 6: Attention Fusion
Build attention/gating network. 1.
Train on classical + deep features. 2.
Evaluate classification performance. 3.
Analyze attention weights by class. 4.
## Output:
Interpretable learned fusion model.
Phase 7: YOLO Hybrid
Extract YOLO FPN features. 1.
Combine YOLO FPN features with classical crop features. 2.
Train hybrid classifier/re-scoring module. 3.
Evaluate detection/classification performance. 4.
## Output:
Final hybrid detection result.
This is the most technically challenging experiment, so it should be done after the main feature-
fusion experiments are working.
## 15. Expected Results
The expected findings should be written carefully, not as guaranteed results.
## Expected Finding 1: Fusion Improves Classification
Classical features may perform well on materials with distinctive color and texture, such as
cardboard and paper.
Deep features may perform better on visually complex materials such as glass, metal, and
plastic.
Therefore, fusion is expected to outperform either feature type alone.
Expected Finding 2: PCA May Preserve Most Performance
PCA is expected to reduce the 2,300-dimensional fused vector significantly while preserving
most classification performance.
This would show that the full fused vector contains redundancy.
Expected Finding 3: Autoencoder May Help at Strong
## Compression
At very low dimensions, autoencoder compression may preserve nonlinear feature relationships
better than PCA.
This is useful if the final system is intended for deployment on limited hardware.
## Expected Finding 4: Attention Fusion May Be More
## Interpretable
The attention model may learn different branch weights for different waste classes.
For example:
ClassPossible Dominant Features
CardboardClassical color/texture
PaperClassical texture/color
GlassDeep visual patterns
MetalDeep + reflectiveness cues
PlasticMixed classical/deep
TrashMixed features
Expected Finding 5: YOLO Hybrid May Improve Class
## Confidence
The YOLO hybrid model may improve classification confidence by adding explicit material cues
to YOLO’s detection features.
However, it may not greatly improve localization because bounding box prediction still comes
mainly from YOLO.
- Main Risks and Solutions
RiskExplanationSolution
Too many experimentsEleven experiments can be time-
consuming
Implement in phases and
keep YOLO hybrid last
Confusion between
detection and classification
Classical/SVM models do not
detect objects by themselves
Use two evaluation tracks
YOLO FPN extraction may be
difficult
Accessing internal YOLO features
requires extra coding
Treat YOLO hybrid as
advanced/final experiment
Dataset imbalanceSome waste classes may have
fewer examples
Use macro-F1 and class
weights
Feature vector too large2,300 dimensions may be
computationally heavy
Use PCA, autoencoder, and
feature selection
OverfittingSome models may memorize
crop-level features
Use validation set, test set,
and fixed split
## 17. Final Deliverables
By the end of the project, the final submission should include:
DeliverableDescription
Dataset preprocessing codeClass merging, splitting, crop extraction
YOLO training resultsDetection baseline
Feature extraction scriptsClassical, EfficientNet, YOLO FPN
Trained classifiersRF, SVM, autoencoder, attention model
Experiment results tableAll 11 experiments compared
Confusion matricesFor classification experiments
Accuracy vs dimension chartShows compression trade-off
Speed comparisonFeature extraction and classification time
Final discussionBest model and explanation
Final report/presentationAcademic project documentation
## 18. Suggested Final Results Table Format
Your final report should include a table like this:
Exp.MethodFeature
## Dim.
## Accurac
y
## Macro-
## F1
mAP@0.5FPSModel
## Size
E1YOLOv8-nanoNative—————
E2Classical + RF252——N/A——
E3EfficientNet +
## SVM
## 1280——N/A——
E4Raw early fusion2300——N/A——
E5PCA fusion~400–600——N/A——
E6Autoencoder
fusion
## 256——N/A——
E7Feature selection200——N/A——
E8Average votingSeparate——N/A——
E9Weighted votingSeparate——N/A——
E10Attention fusion256——N/A——
E11YOLO hybrid768 + 252—————
Use N/Awhere mAP does not apply.
- Suggested Report Structure for Final Submission
When writing the final academic report, use this structure:
## Abstract 1.
## Introduction 2.
## Problem Statement 3.
## Research Question 4.
## Dataset Description 5.
## Preprocessing 6.
## Feature Extraction Methods 7.
## Fusion Strategies 8.
## Dimensionality Reduction Methods 9.
## Experimental Design 10.
## Evaluation Metrics 11.
## Results 12.
## Discussion 13.
## Limitations 14.
## Conclusion 15.
## References 16.
## 20. Final Feasibility Verdict
This project is viable and strong, as long as it is implemented with the correct separation
between detection and classification.
The most important correction is:
YOLO performs localization and detection. Classical, deep, and fused features mainly
evaluate object crop classification and representation quality.
The project should not claim that Random Forest, SVM, PCA, or attention fusion are full object
detectors unless they are integrated into a full detection pipeline with bounding boxes and
confidence scores.
The strongest final version is:
Train YOLOv8-nano as the detection baseline, then conduct a systematic representation
study on waste object crops using classical features, EfficientNet features, fusion methods,
and compression techniques. Finally, test whether YOLO’s detection features can be
improved through a hybrid fusion model.
## Project Description
Thursday, 14 May 20261:59 PM

Comprehensive Pre-Implementation Report
Compression-Aware Multiscale Feature Fusion for Waste
Object Detection and Classification
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
object detection and classificationacross all object sizes.
## 2. Proposed Project Title
## Recommended Title
Compression-Aware Multiscale Feature Fusion for Waste Object Detection and Classification
Alternative title:
Multiscale Feature Fusion and Dimensionality Reduction for Waste Object Detection
The first title is stronger because it clearly shows the two main technical ideas:
Feature fusion 1.
Dimensionality reduction / compression 2.
## 3. Main Research Question
The main research question is:
How can classical visual features, pretrained deep neural features, and YOLO detection
features be fused and compressed to improve waste-object classification accuracy while
reducing feature dimensionality and computational cost?
This question is clear, realistic, and directly connected to Computer Vision.
## 4. Project Objectives
The project has five main objectives:
ObjectiveExplanation
Compare individual feature
types
Test classical, deep, and YOLO-based features separately
Study feature fusionCompare early fusion, late fusion, attention fusion, and YOLO
hybrid fusion
Apply dimensionality
reduction
Use PCA, autoencoder compression, and feature selection
Evaluate accuracy-efficiency
trade-off
Compare accuracy, F1-score, mAP, speed, feature size, and
model size
Identify the best practical
configuration
Recommend the most accurate and efficient representation for
waste detection/classification
## 5. Dataset
The project uses the Kaggle Trash Detection Image Dataset, which contains 4,142 imageswith
bounding box annotations in YOLO format. The original dataset includes seven classes:
Cardboard, Glass, Metal, Paper, Plastic, Trash, and Biodegradable. In the proposed project,
Biodegradable is merged with Trash, producing a final six-class taxonomy.
## Final Class List
Original ClassFinal Class
CardboardCardboard
GlassGlass
MetalMetal
PaperPaper
PlasticPlastic
TrashTrash
BiodegradableTrash
The final classification problem therefore contains six material categories.
## 6. Dataset Split
The dataset will be divided using stratified sampling:
SplitPercentageApproximate Images
## Training70%~2,900
## Validation15%~620
## Test15%~620
The split must be done at the image level, not object level. This prevents objects from the same
image appearing in both training and test sets.
Stratification is important because some classes, especially Metal, may appear less frequently
than others. The split should preserve class distribution as much as possible.
## 7. Important Methodological Clarification
This project contains two related but different tasks:
## Task A: Object Detection
Object detection means the system outputs:
bounding box + class label + confidence score
This applies mainly to:
YOLOv8-nano baseline 1.
YOLO hybrid model 2.
Detection metrics such as mAP@0.5and mAP@0.5:0.95are valid here.
## Task B: Object Crop Classification
Object crop classification means the object location is already known, and the model only
predicts the material class.
This applies to:
Classical feature classifier 1.
Deep feature classifier 2.
Early fusion models 3.
PCA fusion 4.
Autoencoder fusion 5.
Feature-selection fusion 6.
Late fusion 7.
Attention fusion 8.
These models do not perform full detection by themselves. They classify cropped object regions.
So their main metrics should be:
accuracy, macro-F1, per-class F1, feature dimension, extraction time, and classification
time
This distinction is very important. It makes the methodology technically correct.
## 8. General Pipeline
The full project pipeline will follow these stages:
## Stage 1: Dataset Preparation
Load images and YOLO annotation files. 1.
Convert YOLO labels into bounding box coordinates. 2.
Merge Biodegradable into Trash. 3.
Split dataset into train, validation, and test sets. 4.
Crop object regions from images using ground-truth boxes. 5.
Resize crops when needed, especially for EfficientNet. 6.
Stage 2: YOLO Training
A YOLOv8-nano detector will be trained on the six-class dataset.
YOLO will be used for two purposes:
As a detection baseline. 1.
As a region proposal/localization module for hybrid experiments. 2.
## Stage 3: Feature Extraction
For each object crop, three types of features will be extracted:
Classical hand-crafted features 1.
Deep EfficientNetB0 features 2.
YOLO Feature Pyramid Network features 3.
Stage 4: Fusion and Compression
Different fusion strategies will be tested:
Early feature concatenation 1.
PCA dimensionality reduction 2.
Autoencoder compression 3.
Feature selection 4.
Late probability fusion 5.
Attention-based fusion 6.
YOLO hybrid fusion 7.
## Stage 5: Evaluation
The project will evaluate both detection and classification performance.
Detection models will use mAP-based metrics.
Feature/classification models will use accuracy, macro-F1, per-class F1, time, and feature size.
## 9. Feature Extraction Design
## 9.1 Classical Feature Branch
The classical feature branch extracts interpretable visual properties from object crops.
Expected feature size: 252 dimensions
Feature TypeDimensionsPurpose
Edge features5Capture boundary and gradient information
Shape features19Describe object geometry
LBP texture features119Capture local texture patterns
GLCM texture features48Capture texture regularity and direction
HSV color histogram24Capture color distribution
Spatial pyramid features28Preserve coarse spatial layout
Lab perceptual color features7Represent human-perceived color differences
Material-specific cues2Detect highlight/metallic-like properties
Total252Full classical vector
This branch is useful because waste categories often depend on visible material properties such
as color, texture, reflectiveness, and shape.
## 9.2 Deep Hierarchical Feature Branch
The deep feature branch uses a pretrained EfficientNetB0 model.
The model is used as a feature extractor, not as a fully fine-tuned classifier.
Feature LevelDimensionsMeaning
Early-layer features256Edges, corners, colors, low-level texture
Mid-layer features512Material surfaces, local structures
Late-layer features1,280High-level semantic appearance
Total deep features2,048Full EfficientNet representation
The early, mid, and late features capture different levels of visual information.
9.3 YOLO Detection Feature Branch
YOLOv8-nano produces multiscale feature maps through its Feature Pyramid Network.
YOLO Feature MapDimensionsRole
P3256Higher-resolution features
P4256Medium-scale features
P5256Lower-resolution semantic features
Total YOLO FPN features768Detection-specific representation
These features are useful because they are directly optimized for object detection.
- Experiments to Be Conducted
The project should include eleven experimental configurations.
## Experiment Group 1: Baseline Experiments
These experiments establish the performance of each representation type before fusion.
ExperimentNameFeature SourceModelEvaluation Type
E1YOLOv8-nano
baseline
Full image YOLO
features
YOLO detectorDetection
E2Classical-only baseline252 classical featuresRandom
## Forest
## Classification
E3Deep-only baselineEfficientNet late
features
SVMClassification
E1: YOLOv8-Nano Baseline
This experiment trains YOLOv8-nano on the waste dataset.
## Purpose:
Establish the main object detection baseline.
## Outputs:
Bounding boxes •
Class labels •
Confidence scores •
## Metrics:
mAP@0.5 •
mAP@0.5:0.95 •
Per-class AP •
## FPS •
Model size •
E2: Classical-Only Baseline
This experiment extracts 252-dimensional classical features from each object crop and trains a
Random Forest classifier.
## Purpose:
Test how well interpretable visual features classify waste materials.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
E3: Deep-Only Baseline
This experiment extracts EfficientNetB0 late-layer features and trains an SVM classifier.
## Purpose:
Test how well pretrained deep semantic features classify waste objects.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
## Experiment Group 2: Early Fusion Experiments
Early fusion means features are combined first, then one classifier is trained.
## Experime
nt
NameFeatures UsedReduction MethodClassifier
E4Raw early fusionClassical + EfficientNet
early + mid + late
NoneRandom Forest
E5PCA fusionSame fused vectorPCA, 95% varianceSVM
E6Autoencoder
fusion
Same fused vector256-dimensional
bottleneck
Neural classifier
## / SVM
E7Feature-
selection fusion
Same fused vectorTop 200 featuresRandom Forest
The full early-fusion vector is:
Feature SourceDimensions
Classical features252
EfficientNet early256
EfficientNet mid512
EfficientNet late1,280
## Total2,300
## E4: Raw Early Fusion
The 2,300-dimensional fused feature vector is used directly.
## Purpose:
Test whether combining all available classical and deep features improves classification
performance.
Main question:
Does raw feature concatenation outperform individual feature branches?
E5: PCA Fusion
The 2,300-dimensional vector is compressed using PCA while retaining 95% of variance.
## Purpose:
Test whether linear dimensionality reduction can reduce feature size without large
accuracy loss.
Expected output:
Reduced feature vector, probably around 400–600 dimensions •
Classification results after compression •
## E6: Autoencoder Fusion
A shallow autoencoder compresses the 2,300-dimensional vector into a 256-dimensional
bottleneck.
Suggested structure:
## Input 2300 → Dense 512 → Dense 256 → Dense 512 → Output 2300
The 256-dimensional bottleneck is then used for classification.
## Purpose:
Test whether nonlinear compression preserves information better than PCA.
E7: Feature-Selection Fusion
A Random Forest is used to rank feature importance.
The top 200 features are selected and used for classification.
## Purpose:
Test whether only a small subset of the fused features is enough for strong classification
performance.
This experiment is useful because it may reveal which feature groups are most informative.
## Experiment Group 3: Late Fusion Experiments
Late fusion means separate classifiers are trained first, then their predictions are combined.
## Experimen
t
NameModels CombinedFusion Method
E8Average votingClassical classifier + deep
classifier
Equal probability averaging
E9Weighted votingClassical classifier + deep
classifier
Validation-optimized weights
## E8: Average Voting
The classical classifier and deep classifier each produce class probabilities.
The final prediction is:
average of both probability vectors
## Purpose:
Test whether simple decision-level fusion improves performance.
## E9: Weighted Voting
The classical and deep probabilities are combined using optimized weights.
## Example:
Final prediction = 0.4 classical + 0.6 deep
The best weights are selected using the validation set.
## Purpose:
Test whether one representation should be trusted more than another.
This may show that deep features are generally stronger, but classical features still add useful
material-specific information.
Experiment Group 4: Attention-Based Fusion
ExperimentNameFeatures UsedMethod
E10Attention fusionClassical + EfficientNet late featuresLearned gating network
## E10: Attention Fusion
This experiment trains a lightweight neural network that learns how much attention to give to
each feature branch.
General architecture:
Project classical features from 252 dimensions to 256 dimensions. 1.
Project EfficientNet late features from 1,280 dimensions to 256 dimensions. 2.
Concatenate both projected vectors. 3.
Learn attention weights for classical and deep features. 4.
Produce one fused 256-dimensional vector. 5.
Classify the object. 6.
## Purpose:
Test whether the system can dynamically choose which representation is more useful for
each object.
Example interpretation:
Cardboard may rely more on color and texture. •
Glass may rely more on shape, transparency, and deep visual patterns. •
Metal may rely on reflectiveness and deep features. •
This experiment is advanced and gives the project a strong research contribution.
Experiment Group 5: YOLO Hybrid Experiment
## Experimen
t
NameFeatures UsedMethodEvaluation
## E11YOLO
hybrid
YOLO FPN + classical
features
## Fusion/re-scoring
module
## Detection +
classification
E11: YOLO Hybrid
This experiment combines YOLO’s internal detection features with classical hand-crafted
features from the same detected region.
## Purpose:
Test whether YOLO detection features become stronger when combined with explicit
color, texture, and material cues.
Suggested implementation:
Run YOLO on the image. 1.
Extract predicted bounding boxes. 2.
Crop each detected object region. 3.
Extract classical features from the crop. 4.
Extract YOLO FPN features corresponding to the detected region. 5.
Concatenate YOLO FPN features with classical features. 6.
Use a classifier or re-scoring module to refine the class prediction. 7.
Final output:
YOLO bounding box •
Refined class prediction •
Refined confidence score •
Because this system still outputs bounding boxes, class labels, and confidence scores, it can be
evaluated using detection metrics.
## 11. Final Experiment Count
The project contains 11 experiments.
GroupExperimentsCount
BaselinesE1, E2, E33
Early fusionE4, E5, E6, E74
Late fusionE8, E92
Attention fusionE101
YOLO hybridE111
## Total11
So in the report, do not write “thirteen experiments.”
## Use:
eleven experimental configurations
or:
all experimental configurations
## 12. Evaluation Protocol
The evaluation should be divided into two tracks.
## Track A: Detection Evaluation
Used for:
E1: YOLOv8-nano baseline •
E11: YOLO hybrid •
MetricMeaning
mAP@0.5Detection performance at IoU threshold 0.5
mAP@0.5:0.95Stricter detection/localization metric
Per-class APDetection performance for each waste class
PrecisionCorrectness of positive detections
RecallAbility to find existing objects
FPSReal-time speed
Model footprintDeployment size
## Track B: Classification / Representation Evaluation
Used for:
## E2: Classical-only •
## E3: Deep-only •
E4: Raw early fusion •
E5: PCA fusion •
E6: Autoencoder fusion •
E7: Feature-selection fusion •
E8: Average voting •
E9: Weighted voting •
E10: Attention fusion •
MetricMeaning
AccuracyOverall classification correctness
Macro-F1Balanced performance across all classes
Per-class F1Performance for each waste material
Confusion matrixShows which classes are confused
Feature dimensionSize of the representation
Feature extraction timeCost of producing the features
Classification timeCost of predicting the class
Model sizeStorage requirement
Macro-F1 is especially important because the dataset may be imbalanced.
## 13. Controlled Variables
To ensure fair comparison, the following conditions should remain fixed:
Controlled VariableDecision
Dataset splitSame train/validation/test split for all experiments
Class mappingSame six-class taxonomy
Crop generationSame ground-truth crops for classification experiments
Image preprocessingSame resizing and normalization rules
Classifier training dataSame training objects
Test setSame test objects/images
Random seedFixed, for example 42
YOLO confidence thresholdFixed, for example 0.25
NMS IoU thresholdFixed, for example 0.5
## 14. Recommended Implementation Order
To make the project manageable, implement the experiments in this order:
Phase 1: Dataset and YOLO Baseline
Prepare dataset. 1.
Merge classes. 2.
Train YOLOv8-nano. 3.
Evaluate YOLO detection performance. 4.
## Output:
YOLO baseline results.
## Phase 2: Object Crop Dataset
Extract object crops from ground-truth bounding boxes. 1.
Save crop images and labels. 2.
Prepare train/validation/test crop-level datasets. 3.
## Output:
Clean crop classification dataset.
## Phase 3: Baseline Feature Models
Extract classical features. 1.
## Train Random Forest. 2.
Extract EfficientNet features. 3.
Train SVM. 4.
Compare classical-only and deep-only baselines. 5.
## Output:
Baseline representation results.
Phase 4: Early Fusion and Compression
Concatenate classical + EfficientNet features. 1.
Train raw fusion model. 2.
Apply PCA. 3.
Train PCA-fusion classifier. 4.
Train autoencoder. 5.
Extract bottleneck features. 6.
Train autoencoder-fusion classifier. 7.
Perform feature selection. 8.
Train selected-feature classifier. 9.
## Output:
Accuracy vs feature-dimension comparison.
## Phase 5: Late Fusion
Generate probability outputs from classical and deep classifiers. 1.
Apply average voting. 2.
Optimize voting weights on validation set. 3.
Evaluate on test set. 4.
## Output:
Decision-level fusion comparison.
## Phase 6: Attention Fusion
Build attention/gating network. 1.
Train on classical + deep features. 2.
Evaluate classification performance. 3.
Analyze attention weights by class. 4.
## Output:
Interpretable learned fusion model.
Phase 7: YOLO Hybrid
Extract YOLO FPN features. 1.
Combine YOLO FPN features with classical crop features. 2.
Train hybrid classifier/re-scoring module. 3.
Evaluate detection/classification performance. 4.
## Output:
Final hybrid detection result.
This is the most technically challenging experiment, so it should be done after the main feature-
fusion experiments are working.
## 15. Expected Results
The expected findings should be written carefully, not as guaranteed results.
## Expected Finding 1: Fusion Improves Classification
Classical features may perform well on materials with distinctive color and texture, such as
cardboard and paper.
Deep features may perform better on visually complex materials such as glass, metal, and
plastic.
Therefore, fusion is expected to outperform either feature type alone.
Expected Finding 2: PCA May Preserve Most Performance
PCA is expected to reduce the 2,300-dimensional fused vector significantly while preserving
most classification performance.
This would show that the full fused vector contains redundancy.
Expected Finding 3: Autoencoder May Help at Strong
## Compression
At very low dimensions, autoencoder compression may preserve nonlinear feature relationships
better than PCA.
This is useful if the final system is intended for deployment on limited hardware.
## Expected Finding 4: Attention Fusion May Be More
## Interpretable
The attention model may learn different branch weights for different waste classes.
For example:
ClassPossible Dominant Features
CardboardClassical color/texture
PaperClassical texture/color
GlassDeep visual patterns
MetalDeep + reflectiveness cues
PlasticMixed classical/deep
TrashMixed features
Expected Finding 5: YOLO Hybrid May Improve Class
## Confidence
The YOLO hybrid model may improve classification confidence by adding explicit material cues
to YOLO’s detection features.
However, it may not greatly improve localization because bounding box prediction still comes
mainly from YOLO.
- Main Risks and Solutions
RiskExplanationSolution
Too many experimentsEleven experiments can be time-
consuming
Implement in phases and
keep YOLO hybrid last
Confusion between
detection and classification
Classical/SVM models do not
detect objects by themselves
Use two evaluation tracks
YOLO FPN extraction may be
difficult
Accessing internal YOLO features
requires extra coding
Treat YOLO hybrid as
advanced/final experiment
Dataset imbalanceSome waste classes may have
fewer examples
Use macro-F1 and class
weights
Feature vector too large2,300 dimensions may be
computationally heavy
Use PCA, autoencoder, and
feature selection
OverfittingSome models may memorize
crop-level features
Use validation set, test set,
and fixed split
## 17. Final Deliverables
By the end of the project, the final submission should include:
DeliverableDescription
Dataset preprocessing codeClass merging, splitting, crop extraction
YOLO training resultsDetection baseline
Feature extraction scriptsClassical, EfficientNet, YOLO FPN
Trained classifiersRF, SVM, autoencoder, attention model
Experiment results tableAll 11 experiments compared
Confusion matricesFor classification experiments
Accuracy vs dimension chartShows compression trade-off
Speed comparisonFeature extraction and classification time
Final discussionBest model and explanation
Final report/presentationAcademic project documentation
## 18. Suggested Final Results Table Format
Your final report should include a table like this:
Exp.MethodFeature
## Dim.
## Accurac
y
## Macro-
## F1
mAP@0.5FPSModel
## Size
E1YOLOv8-nanoNative—————
E2Classical + RF252——N/A——
E3EfficientNet +
## SVM
## 1280——N/A——
E4Raw early fusion2300——N/A——
E5PCA fusion~400–600——N/A——
E6Autoencoder
fusion
## 256——N/A——
E7Feature selection200——N/A——
E8Average votingSeparate——N/A——
E9Weighted votingSeparate——N/A——
E10Attention fusion256——N/A——
E11YOLO hybrid768 + 252—————
Use N/Awhere mAP does not apply.
- Suggested Report Structure for Final Submission
When writing the final academic report, use this structure:
## Abstract 1.
## Introduction 2.
## Problem Statement 3.
## Research Question 4.
## Dataset Description 5.
## Preprocessing 6.
## Feature Extraction Methods 7.
## Fusion Strategies 8.
## Dimensionality Reduction Methods 9.
## Experimental Design 10.
## Evaluation Metrics 11.
## Results 12.
## Discussion 13.
## Limitations 14.
## Conclusion 15.
## References 16.
## 20. Final Feasibility Verdict
This project is viable and strong, as long as it is implemented with the correct separation
between detection and classification.
The most important correction is:
YOLO performs localization and detection. Classical, deep, and fused features mainly
evaluate object crop classification and representation quality.
The project should not claim that Random Forest, SVM, PCA, or attention fusion are full object
detectors unless they are integrated into a full detection pipeline with bounding boxes and
confidence scores.
The strongest final version is:
Train YOLOv8-nano as the detection baseline, then conduct a systematic representation
study on waste object crops using classical features, EfficientNet features, fusion methods,
and compression techniques. Finally, test whether YOLO’s detection features can be
improved through a hybrid fusion model.
## Project Description
Thursday, 14 May 20261:59 PM

Comprehensive Pre-Implementation Report
Compression-Aware Multiscale Feature Fusion for Waste
Object Detection and Classification
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
object detection and classificationacross all object sizes.
## 2. Proposed Project Title
## Recommended Title
Compression-Aware Multiscale Feature Fusion for Waste Object Detection and Classification
Alternative title:
Multiscale Feature Fusion and Dimensionality Reduction for Waste Object Detection
The first title is stronger because it clearly shows the two main technical ideas:
Feature fusion 1.
Dimensionality reduction / compression 2.
## 3. Main Research Question
The main research question is:
How can classical visual features, pretrained deep neural features, and YOLO detection
features be fused and compressed to improve waste-object classification accuracy while
reducing feature dimensionality and computational cost?
This question is clear, realistic, and directly connected to Computer Vision.
## 4. Project Objectives
The project has five main objectives:
ObjectiveExplanation
Compare individual feature
types
Test classical, deep, and YOLO-based features separately
Study feature fusionCompare early fusion, late fusion, attention fusion, and YOLO
hybrid fusion
Apply dimensionality
reduction
Use PCA, autoencoder compression, and feature selection
Evaluate accuracy-efficiency
trade-off
Compare accuracy, F1-score, mAP, speed, feature size, and
model size
Identify the best practical
configuration
Recommend the most accurate and efficient representation for
waste detection/classification
## 5. Dataset
The project uses the Kaggle Trash Detection Image Dataset, which contains 4,142 imageswith
bounding box annotations in YOLO format. The original dataset includes seven classes:
Cardboard, Glass, Metal, Paper, Plastic, Trash, and Biodegradable. In the proposed project,
Biodegradable is merged with Trash, producing a final six-class taxonomy.
## Final Class List
Original ClassFinal Class
CardboardCardboard
GlassGlass
MetalMetal
PaperPaper
PlasticPlastic
TrashTrash
BiodegradableTrash
The final classification problem therefore contains six material categories.
## 6. Dataset Split
The dataset will be divided using stratified sampling:
SplitPercentageApproximate Images
## Training70%~2,900
## Validation15%~620
## Test15%~620
The split must be done at the image level, not object level. This prevents objects from the same
image appearing in both training and test sets.
Stratification is important because some classes, especially Metal, may appear less frequently
than others. The split should preserve class distribution as much as possible.
## 7. Important Methodological Clarification
This project contains two related but different tasks:
## Task A: Object Detection
Object detection means the system outputs:
bounding box + class label + confidence score
This applies mainly to:
YOLOv8-nano baseline 1.
YOLO hybrid model 2.
Detection metrics such as mAP@0.5and mAP@0.5:0.95are valid here.
## Task B: Object Crop Classification
Object crop classification means the object location is already known, and the model only
predicts the material class.
This applies to:
Classical feature classifier 1.
Deep feature classifier 2.
Early fusion models 3.
PCA fusion 4.
Autoencoder fusion 5.
Feature-selection fusion 6.
Late fusion 7.
Attention fusion 8.
These models do not perform full detection by themselves. They classify cropped object regions.
So their main metrics should be:
accuracy, macro-F1, per-class F1, feature dimension, extraction time, and classification
time
This distinction is very important. It makes the methodology technically correct.
## 8. General Pipeline
The full project pipeline will follow these stages:
## Stage 1: Dataset Preparation
Load images and YOLO annotation files. 1.
Convert YOLO labels into bounding box coordinates. 2.
Merge Biodegradable into Trash. 3.
Split dataset into train, validation, and test sets. 4.
Crop object regions from images using ground-truth boxes. 5.
Resize crops when needed, especially for EfficientNet. 6.
Stage 2: YOLO Training
A YOLOv8-nano detector will be trained on the six-class dataset.
YOLO will be used for two purposes:
As a detection baseline. 1.
As a region proposal/localization module for hybrid experiments. 2.
## Stage 3: Feature Extraction
For each object crop, three types of features will be extracted:
Classical hand-crafted features 1.
Deep EfficientNetB0 features 2.
YOLO Feature Pyramid Network features 3.
Stage 4: Fusion and Compression
Different fusion strategies will be tested:
Early feature concatenation 1.
PCA dimensionality reduction 2.
Autoencoder compression 3.
Feature selection 4.
Late probability fusion 5.
Attention-based fusion 6.
YOLO hybrid fusion 7.
## Stage 5: Evaluation
The project will evaluate both detection and classification performance.
Detection models will use mAP-based metrics.
Feature/classification models will use accuracy, macro-F1, per-class F1, time, and feature size.
## 9. Feature Extraction Design
## 9.1 Classical Feature Branch
The classical feature branch extracts interpretable visual properties from object crops.
Expected feature size: 252 dimensions
Feature TypeDimensionsPurpose
Edge features5Capture boundary and gradient information
Shape features19Describe object geometry
LBP texture features119Capture local texture patterns
GLCM texture features48Capture texture regularity and direction
HSV color histogram24Capture color distribution
Spatial pyramid features28Preserve coarse spatial layout
Lab perceptual color features7Represent human-perceived color differences
Material-specific cues2Detect highlight/metallic-like properties
Total252Full classical vector
This branch is useful because waste categories often depend on visible material properties such
as color, texture, reflectiveness, and shape.
## 9.2 Deep Hierarchical Feature Branch
The deep feature branch uses a pretrained EfficientNetB0 model.
The model is used as a feature extractor, not as a fully fine-tuned classifier.
Feature LevelDimensionsMeaning
Early-layer features256Edges, corners, colors, low-level texture
Mid-layer features512Material surfaces, local structures
Late-layer features1,280High-level semantic appearance
Total deep features2,048Full EfficientNet representation
The early, mid, and late features capture different levels of visual information.
9.3 YOLO Detection Feature Branch
YOLOv8-nano produces multiscale feature maps through its Feature Pyramid Network.
YOLO Feature MapDimensionsRole
P3256Higher-resolution features
P4256Medium-scale features
P5256Lower-resolution semantic features
Total YOLO FPN features768Detection-specific representation
These features are useful because they are directly optimized for object detection.
- Experiments to Be Conducted
The project should include eleven experimental configurations.
## Experiment Group 1: Baseline Experiments
These experiments establish the performance of each representation type before fusion.
## Experimen
t
NameFeature SourceModelEvaluation Type
E1YOLOv8-nano
baseline
Full image YOLO
features
YOLO detectorDetection
E2Classical-only baseline252 classical featuresRandom
## Forest
## Classification
E3Deep-only baselineEfficientNet late
features
SVMClassification
E1: YOLOv8-Nano Baseline
This experiment trains YOLOv8-nano on the waste dataset.
## Purpose:
Establish the main object detection baseline.
## Outputs:
Bounding boxes •
Class labels •
Confidence scores •
## Metrics:
mAP@0.5 •
mAP@0.5:0.95 •
Per-class AP •
## FPS •
Model size •
E2: Classical-Only Baseline
This experiment extracts 252-dimensional classical features from each object crop and trains a
Random Forest classifier.
## Purpose:
Test how well interpretable visual features classify waste materials.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
E3: Deep-Only Baseline
This experiment extracts EfficientNetB0 late-layer features and trains an SVM classifier.
## Purpose:
Test how well pretrained deep semantic features classify waste objects.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
## Experiment Group 2: Early Fusion Experiments
Early fusion means features are combined first, then one classifier is trained.
## Experime
nt
NameFeatures UsedReduction MethodClassifier
E4Raw early fusionClassical + EfficientNet
early + mid + late
NoneRandom Forest
E5PCA fusionSame fused vectorPCA, 95% varianceSVM
E6Autoencoder
fusion
Same fused vector256-dimensional
bottleneck
Neural classifier
## / SVM
E7Feature-
selection fusion
Same fused vectorTop 200 featuresRandom Forest
The full early-fusion vector is:
Feature SourceDimensions
Classical features252
EfficientNet early256
EfficientNet mid512
EfficientNet late1,280
## Total2,300
## E4: Raw Early Fusion
The 2,300-dimensional fused feature vector is used directly.
## Purpose:
Test whether combining all available classical and deep features improves classification
performance.
Main question:
Does raw feature concatenation outperform individual feature branches?
E5: PCA Fusion
The 2,300-dimensional vector is compressed using PCA while retaining 95% of variance.
## Purpose:
Test whether linear dimensionality reduction can reduce feature size without large
accuracy loss.
Expected output:
Reduced feature vector, probably around 400–600 dimensions •
Classification results after compression •
## E6: Autoencoder Fusion
A shallow autoencoder compresses the 2,300-dimensional vector into a 256-dimensional
bottleneck.
Suggested structure:
## Input 2300 → Dense 512 → Dense 256 → Dense 512 → Output 2300
The 256-dimensional bottleneck is then used for classification.
## Purpose:
Test whether nonlinear compression preserves information better than PCA.
E7: Feature-Selection Fusion
A Random Forest is used to rank feature importance.
The top 200 features are selected and used for classification.
## Purpose:
Test whether only a small subset of the fused features is enough for strong classification
performance.
This experiment is useful because it may reveal which feature groups are most informative.
## Experiment Group 3: Late Fusion Experiments
Late fusion means separate classifiers are trained first, then their predictions are combined.
ExperimentNameModels CombinedFusion Method
E8Average votingClassical classifier + deep
classifier
Equal probability averaging
E9Weighted votingClassical classifier + deep
classifier
## Validation-optimized
weights
## E8: Average Voting
The classical classifier and deep classifier each produce class probabilities.
The final prediction is:
average of both probability vectors
## Purpose:
Test whether simple decision-level fusion improves performance.
## E9: Weighted Voting
The classical and deep probabilities are combined using optimized weights.
## Example:
Final prediction = 0.4 classical + 0.6 deep
The best weights are selected using the validation set.
## Purpose:
Test whether one representation should be trusted more than another.
This may show that deep features are generally stronger, but classical features still add useful
material-specific information.
Experiment Group 4: Attention-Based Fusion
ExperimentNameFeatures UsedMethod
E10Attention fusionClassical + EfficientNet late featuresLearned gating network
## E10: Attention Fusion
This experiment trains a lightweight neural network that learns how much attention to give to
each feature branch.
General architecture:
Project classical features from 252 dimensions to 256 dimensions. 1.
Project EfficientNet late features from 1,280 dimensions to 256 dimensions. 2.
Concatenate both projected vectors. 3.
Learn attention weights for classical and deep features. 4.
Produce one fused 256-dimensional vector. 5.
Classify the object. 6.
## Purpose:
Test whether the system can dynamically choose which representation is more useful for
each object.
Example interpretation:
Cardboard may rely more on color and texture. •
Glass may rely more on shape, transparency, and deep visual patterns. •
Metal may rely on reflectiveness and deep features. •
This experiment is advanced and gives the project a strong research contribution.
Experiment Group 5: YOLO Hybrid Experiment
## Experimen
t
NameFeatures UsedMethodEvaluation
## E11YOLO
hybrid
YOLO FPN + classical
features
## Fusion/re-scoring
module
## Detection +
classification
E11: YOLO Hybrid
This experiment combines YOLO’s internal detection features with classical hand-crafted
features from the same detected region.
## Purpose:
Test whether YOLO detection features become stronger when combined with explicit
color, texture, and material cues.
Suggested implementation:
Run YOLO on the image. 1.
Extract predicted bounding boxes. 2.
Crop each detected object region. 3.
Extract classical features from the crop. 4.
Extract YOLO FPN features corresponding to the detected region. 5.
Concatenate YOLO FPN features with classical features. 6.
Use a classifier or re-scoring module to refine the class prediction. 7.
Final output:
YOLO bounding box •
Refined class prediction •
Refined confidence score •
Because this system still outputs bounding boxes, class labels, and confidence scores, it can be
evaluated using detection metrics.
## 11. Final Experiment Count
The project contains 11 experiments.
GroupExperimentsCount
BaselinesE1, E2, E33
Early fusionE4, E5, E6, E74
Late fusionE8, E92
Attention fusionE101
YOLO hybridE111
## Total11
So in the report, do not write “thirteen experiments.”
## Use:
eleven experimental configurations
or:
all experimental configurations
## 12. Evaluation Protocol
The evaluation should be divided into two tracks.
## Track A: Detection Evaluation
Used for:
E1: YOLOv8-nano baseline •
E11: YOLO hybrid •
MetricMeaning
mAP@0.5Detection performance at IoU threshold 0.5
mAP@0.5:0.95Stricter detection/localization metric
Per-class APDetection performance for each waste class
PrecisionCorrectness of positive detections
RecallAbility to find existing objects
FPSReal-time speed
Model footprintDeployment size
## Track B: Classification / Representation Evaluation
Used for:
## E2: Classical-only •
## E3: Deep-only •
E4: Raw early fusion •
E5: PCA fusion •
E6: Autoencoder fusion •
E7: Feature-selection fusion •
E8: Average voting •
E9: Weighted voting •
E10: Attention fusion •
MetricMeaning
AccuracyOverall classification correctness
Macro-F1Balanced performance across all classes
Per-class F1Performance for each waste material
Confusion matrixShows which classes are confused
Feature dimensionSize of the representation
Feature extraction timeCost of producing the features
Classification timeCost of predicting the class
Model sizeStorage requirement
Macro-F1 is especially important because the dataset may be imbalanced.
## 13. Controlled Variables
To ensure fair comparison, the following conditions should remain fixed:
Controlled VariableDecision
Dataset splitSame train/validation/test split for all experiments
Class mappingSame six-class taxonomy
Crop generationSame ground-truth crops for classification experiments
Image preprocessingSame resizing and normalization rules
Classifier training dataSame training objects
Test setSame test objects/images
Random seedFixed, for example 42
YOLO confidence thresholdFixed, for example 0.25
NMS IoU thresholdFixed, for example 0.5
## 14. Recommended Implementation Order
To make the project manageable, implement the experiments in this order:
Phase 1: Dataset and YOLO Baseline
Prepare dataset. 1.
Merge classes. 2.
Train YOLOv8-nano. 3.
Evaluate YOLO detection performance. 4.
## Output:
YOLO baseline results.
## Phase 2: Object Crop Dataset
Extract object crops from ground-truth bounding boxes. 1.
Save crop images and labels. 2.
Prepare train/validation/test crop-level datasets. 3.
## Output:
Clean crop classification dataset.
## Phase 3: Baseline Feature Models
Extract classical features. 1.
## Train Random Forest. 2.
Extract EfficientNet features. 3.
Train SVM. 4.
Compare classical-only and deep-only baselines. 5.
## Output:
Baseline representation results.
Phase 4: Early Fusion and Compression
Concatenate classical + EfficientNet features. 1.
Train raw fusion model. 2.
Apply PCA. 3.
Train PCA-fusion classifier. 4.
Train autoencoder. 5.
Extract bottleneck features. 6.
Train autoencoder-fusion classifier. 7.
Perform feature selection. 8.
Train selected-feature classifier. 9.
## Output:
Accuracy vs feature-dimension comparison.
## Phase 5: Late Fusion
Generate probability outputs from classical and deep classifiers. 1.
Apply average voting. 2.
Optimize voting weights on validation set. 3.
Evaluate on test set. 4.
## Output:
Decision-level fusion comparison.
## Phase 6: Attention Fusion
Build attention/gating network. 1.
Train on classical + deep features. 2.
Evaluate classification performance. 3.
Analyze attention weights by class. 4.
## Output:
Interpretable learned fusion model.
Phase 7: YOLO Hybrid
Extract YOLO FPN features. 1.
Combine YOLO FPN features with classical crop features. 2.
Train hybrid classifier/re-scoring module. 3.
Evaluate detection/classification performance. 4.
## Output:
Final hybrid detection result.
This is the most technically challenging experiment, so it should be done after the main feature-
fusion experiments are working.
## 15. Expected Results
The expected findings should be written carefully, not as guaranteed results.
## Expected Finding 1: Fusion Improves Classification
Classical features may perform well on materials with distinctive color and texture, such as
cardboard and paper.
Deep features may perform better on visually complex materials such as glass, metal, and
plastic.
Therefore, fusion is expected to outperform either feature type alone.
Expected Finding 2: PCA May Preserve Most Performance
PCA is expected to reduce the 2,300-dimensional fused vector significantly while preserving
most classification performance.
This would show that the full fused vector contains redundancy.
Expected Finding 3: Autoencoder May Help at Strong
## Compression
At very low dimensions, autoencoder compression may preserve nonlinear feature relationships
better than PCA.
This is useful if the final system is intended for deployment on limited hardware.
## Expected Finding 4: Attention Fusion May Be More
## Interpretable
The attention model may learn different branch weights for different waste classes.
For example:
ClassPossible Dominant Features
CardboardClassical color/texture
PaperClassical texture/color
GlassDeep visual patterns
MetalDeep + reflectiveness cues
PlasticMixed classical/deep
TrashMixed features
Expected Finding 5: YOLO Hybrid May Improve Class
## Confidence
The YOLO hybrid model may improve classification confidence by adding explicit material cues
to YOLO’s detection features.
However, it may not greatly improve localization because bounding box prediction still comes
mainly from YOLO.
- Main Risks and Solutions
RiskExplanationSolution
Too many experimentsEleven experiments can be time-
consuming
Implement in phases and
keep YOLO hybrid last
Confusion between
detection and classification
Classical/SVM models do not
detect objects by themselves
Use two evaluation tracks
YOLO FPN extraction may be
difficult
Accessing internal YOLO features
requires extra coding
Treat YOLO hybrid as
advanced/final experiment
Dataset imbalanceSome waste classes may have
fewer examples
Use macro-F1 and class
weights
Feature vector too large2,300 dimensions may be
computationally heavy
Use PCA, autoencoder, and
feature selection
OverfittingSome models may memorize
crop-level features
Use validation set, test set,
and fixed split
## 17. Final Deliverables
By the end of the project, the final submission should include:
DeliverableDescription
Dataset preprocessing codeClass merging, splitting, crop extraction
YOLO training resultsDetection baseline
Feature extraction scriptsClassical, EfficientNet, YOLO FPN
Trained classifiersRF, SVM, autoencoder, attention model
Experiment results tableAll 11 experiments compared
Confusion matricesFor classification experiments
Accuracy vs dimension chartShows compression trade-off
Speed comparisonFeature extraction and classification time
Final discussionBest model and explanation
Final report/presentationAcademic project documentation
## 18. Suggested Final Results Table Format
Your final report should include a table like this:
Exp.MethodFeature
## Dim.
## Accurac
y
## Macro-
## F1
mAP@0.5FPSModel
## Size
E1YOLOv8-nanoNative—————
E2Classical + RF252——N/A——
E3EfficientNet +
## SVM
## 1280——N/A——
E4Raw early fusion2300——N/A——
E5PCA fusion~400–600——N/A——
E6Autoencoder
fusion
## 256——N/A——
E7Feature selection200——N/A——
E8Average votingSeparate——N/A——
E9Weighted votingSeparate——N/A——
E10Attention fusion256——N/A——
E11YOLO hybrid768 + 252—————
Use N/Awhere mAP does not apply.
- Suggested Report Structure for Final Submission
When writing the final academic report, use this structure:
## Abstract 1.
## Introduction 2.
## Problem Statement 3.
## Research Question 4.
## Dataset Description 5.
## Preprocessing 6.
## Feature Extraction Methods 7.
## Fusion Strategies 8.
## Dimensionality Reduction Methods 9.
## Experimental Design 10.
## Evaluation Metrics 11.
## Results 12.
## Discussion 13.
## Limitations 14.
## Conclusion 15.
## References 16.
## 20. Final Feasibility Verdict
This project is viable and strong, as long as it is implemented with the correct separation
between detection and classification.
The most important correction is:
YOLO performs localization and detection. Classical, deep, and fused features mainly
evaluate object crop classification and representation quality.
The project should not claim that Random Forest, SVM, PCA, or attention fusion are full object
detectors unless they are integrated into a full detection pipeline with bounding boxes and
confidence scores.
The strongest final version is:
Train YOLOv8-nano as the detection baseline, then conduct a systematic representation
study on waste object crops using classical features, EfficientNet features, fusion methods,
and compression techniques. Finally, test whether YOLO’s detection features can be
improved through a hybrid fusion model.
## Project Description
Thursday, 14 May 20261:59 PM

Comprehensive Pre-Implementation Report
Compression-Aware Multiscale Feature Fusion for Waste
Object Detection and Classification
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
object detection and classificationacross all object sizes.
## 2. Proposed Project Title
## Recommended Title
Compression-Aware Multiscale Feature Fusion for Waste Object Detection and Classification
Alternative title:
Multiscale Feature Fusion and Dimensionality Reduction for Waste Object Detection
The first title is stronger because it clearly shows the two main technical ideas:
Feature fusion 1.
Dimensionality reduction / compression 2.
## 3. Main Research Question
The main research question is:
How can classical visual features, pretrained deep neural features, and YOLO detection
features be fused and compressed to improve waste-object classification accuracy while
reducing feature dimensionality and computational cost?
This question is clear, realistic, and directly connected to Computer Vision.
## 4. Project Objectives
The project has five main objectives:
ObjectiveExplanation
Compare individual feature
types
Test classical, deep, and YOLO-based features separately
Study feature fusionCompare early fusion, late fusion, attention fusion, and YOLO
hybrid fusion
Apply dimensionality
reduction
Use PCA, autoencoder compression, and feature selection
Evaluate accuracy-efficiency
trade-off
Compare accuracy, F1-score, mAP, speed, feature size, and
model size
Identify the best practical
configuration
Recommend the most accurate and efficient representation for
waste detection/classification
## 5. Dataset
The project uses the Kaggle Trash Detection Image Dataset, which contains 4,142 imageswith
bounding box annotations in YOLO format. The original dataset includes seven classes:
Cardboard, Glass, Metal, Paper, Plastic, Trash, and Biodegradable. In the proposed project,
Biodegradable is merged with Trash, producing a final six-class taxonomy.
## Final Class List
Original ClassFinal Class
CardboardCardboard
GlassGlass
MetalMetal
PaperPaper
PlasticPlastic
TrashTrash
BiodegradableTrash
The final classification problem therefore contains six material categories.
## 6. Dataset Split
The dataset will be divided using stratified sampling:
SplitPercentageApproximate Images
## Training70%~2,900
## Validation15%~620
## Test15%~620
The split must be done at the image level, not object level. This prevents objects from the same
image appearing in both training and test sets.
Stratification is important because some classes, especially Metal, may appear less frequently
than others. The split should preserve class distribution as much as possible.
## 7. Important Methodological Clarification
This project contains two related but different tasks:
## Task A: Object Detection
Object detection means the system outputs:
bounding box + class label + confidence score
This applies mainly to:
YOLOv8-nano baseline 1.
YOLO hybrid model 2.
Detection metrics such as mAP@0.5and mAP@0.5:0.95are valid here.
## Task B: Object Crop Classification
Object crop classification means the object location is already known, and the model only
predicts the material class.
This applies to:
Classical feature classifier 1.
Deep feature classifier 2.
Early fusion models 3.
PCA fusion 4.
Autoencoder fusion 5.
Feature-selection fusion 6.
Late fusion 7.
Attention fusion 8.
These models do not perform full detection by themselves. They classify cropped object regions.
So their main metrics should be:
accuracy, macro-F1, per-class F1, feature dimension, extraction time, and classification
time
This distinction is very important. It makes the methodology technically correct.
## 8. General Pipeline
The full project pipeline will follow these stages:
## Stage 1: Dataset Preparation
Load images and YOLO annotation files. 1.
Convert YOLO labels into bounding box coordinates. 2.
Merge Biodegradable into Trash. 3.
Split dataset into train, validation, and test sets. 4.
Crop object regions from images using ground-truth boxes. 5.
Resize crops when needed, especially for EfficientNet. 6.
Stage 2: YOLO Training
A YOLOv8-nano detector will be trained on the six-class dataset.
YOLO will be used for two purposes:
As a detection baseline. 1.
As a region proposal/localization module for hybrid experiments. 2.
## Stage 3: Feature Extraction
For each object crop, three types of features will be extracted:
Classical hand-crafted features 1.
Deep EfficientNetB0 features 2.
YOLO Feature Pyramid Network features 3.
Stage 4: Fusion and Compression
Different fusion strategies will be tested:
Early feature concatenation 1.
PCA dimensionality reduction 2.
Autoencoder compression 3.
Feature selection 4.
Late probability fusion 5.
Attention-based fusion 6.
YOLO hybrid fusion 7.
## Stage 5: Evaluation
The project will evaluate both detection and classification performance.
Detection models will use mAP-based metrics.
Feature/classification models will use accuracy, macro-F1, per-class F1, time, and feature size.
## 9. Feature Extraction Design
## 9.1 Classical Feature Branch
The classical feature branch extracts interpretable visual properties from object crops.
Expected feature size: 252 dimensions
Feature TypeDimensionsPurpose
Edge features5Capture boundary and gradient information
Shape features19Describe object geometry
LBP texture features119Capture local texture patterns
GLCM texture features48Capture texture regularity and direction
HSV color histogram24Capture color distribution
Spatial pyramid features28Preserve coarse spatial layout
Lab perceptual color features7Represent human-perceived color differences
Material-specific cues2Detect highlight/metallic-like properties
Total252Full classical vector
This branch is useful because waste categories often depend on visible material properties such
as color, texture, reflectiveness, and shape.
## 9.2 Deep Hierarchical Feature Branch
The deep feature branch uses a pretrained EfficientNetB0 model.
The model is used as a feature extractor, not as a fully fine-tuned classifier.
Feature LevelDimensionsMeaning
Early-layer features256Edges, corners, colors, low-level texture
Mid-layer features512Material surfaces, local structures
Late-layer features1,280High-level semantic appearance
Total deep features2,048Full EfficientNet representation
The early, mid, and late features capture different levels of visual information.
9.3 YOLO Detection Feature Branch
YOLOv8-nano produces multiscale feature maps through its Feature Pyramid Network.
YOLO Feature MapDimensionsRole
P3256Higher-resolution features
P4256Medium-scale features
P5256Lower-resolution semantic features
Total YOLO FPN features768Detection-specific representation
These features are useful because they are directly optimized for object detection.
- Experiments to Be Conducted
The project should include eleven experimental configurations.
## Experiment Group 1: Baseline Experiments
These experiments establish the performance of each representation type before fusion.
ExperimentNameFeature SourceModelEvaluation Type
E1YOLOv8-nano
baseline
Full image YOLO
features
YOLO detectorDetection
E2Classical-only baseline252 classical featuresRandom
## Forest
## Classification
E3Deep-only baselineEfficientNet late
features
SVMClassification
E1: YOLOv8-Nano Baseline
This experiment trains YOLOv8-nano on the waste dataset.
## Purpose:
Establish the main object detection baseline.
## Outputs:
Bounding boxes •
Class labels •
Confidence scores •
## Metrics:
mAP@0.5 •
mAP@0.5:0.95 •
Per-class AP •
## FPS •
Model size •
E2: Classical-Only Baseline
This experiment extracts 252-dimensional classical features from each object crop and trains a
Random Forest classifier.
## Purpose:
Test how well interpretable visual features classify waste materials.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
E3: Deep-Only Baseline
This experiment extracts EfficientNetB0 late-layer features and trains an SVM classifier.
## Purpose:
Test how well pretrained deep semantic features classify waste objects.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
## Experiment Group 2: Early Fusion Experiments
Early fusion means features are combined first, then one classifier is trained.
## Experime
nt
NameFeatures UsedReduction MethodClassifier
E4Raw early fusionClassical + EfficientNet
early + mid + late
NoneRandom Forest
E5PCA fusionSame fused vectorPCA, 95% varianceSVM
E6Autoencoder
fusion
Same fused vector256-dimensional
bottleneck
Neural classifier
## / SVM
E7Feature-
selection fusion
Same fused vectorTop 200 featuresRandom Forest
The full early-fusion vector is:
Feature SourceDimensions
Classical features252
EfficientNet early256
EfficientNet mid512
EfficientNet late1,280
## Total2,300
## E4: Raw Early Fusion
The 2,300-dimensional fused feature vector is used directly.
## Purpose:
Test whether combining all available classical and deep features improves classification
performance.
Main question:
Does raw feature concatenation outperform individual feature branches?
E5: PCA Fusion
The 2,300-dimensional vector is compressed using PCA while retaining 95% of variance.
## Purpose:
Test whether linear dimensionality reduction can reduce feature size without large
accuracy loss.
Expected output:
Reduced feature vector, probably around 400–600 dimensions •
Classification results after compression •
## E6: Autoencoder Fusion
A shallow autoencoder compresses the 2,300-dimensional vector into a 256-dimensional
bottleneck.
Suggested structure:
## Input 2300 → Dense 512 → Dense 256 → Dense 512 → Output 2300
The 256-dimensional bottleneck is then used for classification.
## Purpose:
Test whether nonlinear compression preserves information better than PCA.
E7: Feature-Selection Fusion
A Random Forest is used to rank feature importance.
The top 200 features are selected and used for classification.
## Purpose:
Test whether only a small subset of the fused features is enough for strong classification
performance.
This experiment is useful because it may reveal which feature groups are most informative.
## Experiment Group 3: Late Fusion Experiments
Late fusion means separate classifiers are trained first, then their predictions are combined.
ExperimentNameModels CombinedFusion Method
E8Average votingClassical classifier + deep
classifier
Equal probability averaging
E9Weighted
voting
Classical classifier + deep
classifier
Validation-optimized weights
## E8: Average Voting
The classical classifier and deep classifier each produce class probabilities.
The final prediction is:
average of both probability vectors
## Purpose:
Test whether simple decision-level fusion improves performance.
## E9: Weighted Voting
The classical and deep probabilities are combined using optimized weights.
## Example:
Final prediction = 0.4 classical + 0.6 deep
The best weights are selected using the validation set.
## Purpose:
Test whether one representation should be trusted more than another.
This may show that deep features are generally stronger, but classical features still add useful
material-specific information.
Experiment Group 4: Attention-Based Fusion
ExperimentNameFeatures UsedMethod
E10Attention fusionClassical + EfficientNet late featuresLearned gating network
## E10: Attention Fusion
This experiment trains a lightweight neural network that learns how much attention to give to
each feature branch.
General architecture:
Project classical features from 252 dimensions to 256 dimensions. 1.
Project EfficientNet late features from 1,280 dimensions to 256 dimensions. 2.
Concatenate both projected vectors. 3.
Learn attention weights for classical and deep features. 4.
Produce one fused 256-dimensional vector. 5.
Classify the object. 6.
## Purpose:
Test whether the system can dynamically choose which representation is more useful for
each object.
Example interpretation:
Cardboard may rely more on color and texture. •
Glass may rely more on shape, transparency, and deep visual patterns. •
Metal may rely on reflectiveness and deep features. •
This experiment is advanced and gives the project a strong research contribution.
Experiment Group 5: YOLO Hybrid Experiment
## Experimen
t
NameFeatures UsedMethodEvaluation
## E11YOLO
hybrid
YOLO FPN + classical
features
## Fusion/re-scoring
module
## Detection +
classification
E11: YOLO Hybrid
This experiment combines YOLO’s internal detection features with classical hand-crafted
features from the same detected region.
## Purpose:
Test whether YOLO detection features become stronger when combined with explicit
color, texture, and material cues.
Suggested implementation:
Run YOLO on the image. 1.
Extract predicted bounding boxes. 2.
Crop each detected object region. 3.
Extract classical features from the crop. 4.
Extract YOLO FPN features corresponding to the detected region. 5.
Concatenate YOLO FPN features with classical features. 6.
Use a classifier or re-scoring module to refine the class prediction. 7.
Final output:
YOLO bounding box •
Refined class prediction •
Refined confidence score •
Because this system still outputs bounding boxes, class labels, and confidence scores, it can be
evaluated using detection metrics.
## 11. Final Experiment Count
The project contains 11 experiments.
GroupExperimentsCount
BaselinesE1, E2, E33
Early fusionE4, E5, E6, E74
Late fusionE8, E92
Attention fusionE101
YOLO hybridE111
## Total11
So in the report, do not write “thirteen experiments.”
## Use:
eleven experimental configurations
or:
all experimental configurations
## 12. Evaluation Protocol
The evaluation should be divided into two tracks.
## Track A: Detection Evaluation
Used for:
E1: YOLOv8-nano baseline •
E11: YOLO hybrid •
MetricMeaning
mAP@0.5Detection performance at IoU threshold 0.5
mAP@0.5:0.95Stricter detection/localization metric
Per-class APDetection performance for each waste class
PrecisionCorrectness of positive detections
RecallAbility to find existing objects
FPSReal-time speed
Model footprintDeployment size
## Track B: Classification / Representation Evaluation
Used for:
## E2: Classical-only •
## E3: Deep-only •
E4: Raw early fusion •
E5: PCA fusion •
E6: Autoencoder fusion •
E7: Feature-selection fusion •
E8: Average voting •
E9: Weighted voting •
E10: Attention fusion •
MetricMeaning
AccuracyOverall classification correctness
Macro-F1Balanced performance across all classes
Per-class F1Performance for each waste material
Confusion matrixShows which classes are confused
Feature dimensionSize of the representation
Feature extraction timeCost of producing the features
Classification timeCost of predicting the class
Model sizeStorage requirement
Macro-F1 is especially important because the dataset may be imbalanced.
## 13. Controlled Variables
To ensure fair comparison, the following conditions should remain fixed:
Controlled VariableDecision
Dataset splitSame train/validation/test split for all experiments
Class mappingSame six-class taxonomy
Crop generationSame ground-truth crops for classification experiments
Image preprocessingSame resizing and normalization rules
Classifier training dataSame training objects
Test setSame test objects/images
Random seedFixed, for example 42
YOLO confidence thresholdFixed, for example 0.25
NMS IoU thresholdFixed, for example 0.5
## 14. Recommended Implementation Order
To make the project manageable, implement the experiments in this order:
Phase 1: Dataset and YOLO Baseline
Prepare dataset. 1.
Merge classes. 2.
Train YOLOv8-nano. 3.
Evaluate YOLO detection performance. 4.
## Output:
YOLO baseline results.
## Phase 2: Object Crop Dataset
Extract object crops from ground-truth bounding boxes. 1.
Save crop images and labels. 2.
Prepare train/validation/test crop-level datasets. 3.
## Output:
Clean crop classification dataset.
## Phase 3: Baseline Feature Models
Extract classical features. 1.
## Train Random Forest. 2.
Extract EfficientNet features. 3.
Train SVM. 4.
Compare classical-only and deep-only baselines. 5.
## Output:
Baseline representation results.
Phase 4: Early Fusion and Compression
Concatenate classical + EfficientNet features. 1.
Train raw fusion model. 2.
Apply PCA. 3.
Train PCA-fusion classifier. 4.
Train autoencoder. 5.
Extract bottleneck features. 6.
Train autoencoder-fusion classifier. 7.
Perform feature selection. 8.
Train selected-feature classifier. 9.
## Output:
Accuracy vs feature-dimension comparison.
## Phase 5: Late Fusion
Generate probability outputs from classical and deep classifiers. 1.
Apply average voting. 2.
Optimize voting weights on validation set. 3.
Evaluate on test set. 4.
## Output:
Decision-level fusion comparison.
## Phase 6: Attention Fusion
Build attention/gating network. 1.
Train on classical + deep features. 2.
Evaluate classification performance. 3.
Analyze attention weights by class. 4.
## Output:
Interpretable learned fusion model.
Phase 7: YOLO Hybrid
Extract YOLO FPN features. 1.
Combine YOLO FPN features with classical crop features. 2.
Train hybrid classifier/re-scoring module. 3.
Evaluate detection/classification performance. 4.
## Output:
Final hybrid detection result.
This is the most technically challenging experiment, so it should be done after the main feature-
fusion experiments are working.
## 15. Expected Results
The expected findings should be written carefully, not as guaranteed results.
## Expected Finding 1: Fusion Improves Classification
Classical features may perform well on materials with distinctive color and texture, such as
cardboard and paper.
Deep features may perform better on visually complex materials such as glass, metal, and
plastic.
Therefore, fusion is expected to outperform either feature type alone.
Expected Finding 2: PCA May Preserve Most Performance
PCA is expected to reduce the 2,300-dimensional fused vector significantly while preserving
most classification performance.
This would show that the full fused vector contains redundancy.
Expected Finding 3: Autoencoder May Help at Strong
## Compression
At very low dimensions, autoencoder compression may preserve nonlinear feature relationships
better than PCA.
This is useful if the final system is intended for deployment on limited hardware.
## Expected Finding 4: Attention Fusion May Be More
## Interpretable
The attention model may learn different branch weights for different waste classes.
For example:
ClassPossible Dominant Features
CardboardClassical color/texture
PaperClassical texture/color
GlassDeep visual patterns
MetalDeep + reflectiveness cues
PlasticMixed classical/deep
TrashMixed features
Expected Finding 5: YOLO Hybrid May Improve Class
## Confidence
The YOLO hybrid model may improve classification confidence by adding explicit material cues
to YOLO’s detection features.
However, it may not greatly improve localization because bounding box prediction still comes
mainly from YOLO.
- Main Risks and Solutions
RiskExplanationSolution
Too many experimentsEleven experiments can be time-
consuming
Implement in phases and
keep YOLO hybrid last
Confusion between
detection and classification
Classical/SVM models do not
detect objects by themselves
Use two evaluation tracks
YOLO FPN extraction may be
difficult
Accessing internal YOLO features
requires extra coding
Treat YOLO hybrid as
advanced/final experiment
Dataset imbalanceSome waste classes may have
fewer examples
Use macro-F1 and class
weights
Feature vector too large2,300 dimensions may be
computationally heavy
Use PCA, autoencoder, and
feature selection
OverfittingSome models may memorize
crop-level features
Use validation set, test set,
and fixed split
## 17. Final Deliverables
By the end of the project, the final submission should include:
DeliverableDescription
Dataset preprocessing codeClass merging, splitting, crop extraction
YOLO training resultsDetection baseline
Feature extraction scriptsClassical, EfficientNet, YOLO FPN
Trained classifiersRF, SVM, autoencoder, attention model
Experiment results tableAll 11 experiments compared
Confusion matricesFor classification experiments
Accuracy vs dimension chartShows compression trade-off
Speed comparisonFeature extraction and classification time
Final discussionBest model and explanation
Final report/presentationAcademic project documentation
## 18. Suggested Final Results Table Format
Your final report should include a table like this:
Exp.MethodFeature
## Dim.
## Accurac
y
## Macro-
## F1
mAP@0.5FPSModel
## Size
E1YOLOv8-nanoNative—————
E2Classical + RF252——N/A——
E3EfficientNet +
## SVM
## 1280——N/A——
E4Raw early fusion2300——N/A——
E5PCA fusion~400–600——N/A——
E6Autoencoder
fusion
## 256——N/A——
E7Feature selection200——N/A——
E8Average votingSeparate——N/A——
E9Weighted votingSeparate——N/A——
E10Attention fusion256——N/A——
E11YOLO hybrid768 + 252—————
Use N/Awhere mAP does not apply.
- Suggested Report Structure for Final Submission
When writing the final academic report, use this structure:
## Abstract 1.
## Introduction 2.
## Problem Statement 3.
## Research Question 4.
## Dataset Description 5.
## Preprocessing 6.
## Feature Extraction Methods 7.
## Fusion Strategies 8.
## Dimensionality Reduction Methods 9.
## Experimental Design 10.
## Evaluation Metrics 11.
## Results 12.
## Discussion 13.
## Limitations 14.
## Conclusion 15.
## References 16.
## 20. Final Feasibility Verdict
This project is viable and strong, as long as it is implemented with the correct separation
between detection and classification.
The most important correction is:
YOLO performs localization and detection. Classical, deep, and fused features mainly
evaluate object crop classification and representation quality.
The project should not claim that Random Forest, SVM, PCA, or attention fusion are full object
detectors unless they are integrated into a full detection pipeline with bounding boxes and
confidence scores.
The strongest final version is:
Train YOLOv8-nano as the detection baseline, then conduct a systematic representation
study on waste object crops using classical features, EfficientNet features, fusion methods,
and compression techniques. Finally, test whether YOLO’s detection features can be
improved through a hybrid fusion model.
## Project Description
Thursday, 14 May 20261:59 PM

Comprehensive Pre-Implementation Report
Compression-Aware Multiscale Feature Fusion for Waste
Object Detection and Classification
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
object detection and classificationacross all object sizes.
## 2. Proposed Project Title
## Recommended Title
Compression-Aware Multiscale Feature Fusion for Waste Object Detection and Classification
Alternative title:
Multiscale Feature Fusion and Dimensionality Reduction for Waste Object Detection
The first title is stronger because it clearly shows the two main technical ideas:
Feature fusion 1.
Dimensionality reduction / compression 2.
## 3. Main Research Question
The main research question is:
How can classical visual features, pretrained deep neural features, and YOLO detection
features be fused and compressed to improve waste-object classification accuracy while
reducing feature dimensionality and computational cost?
This question is clear, realistic, and directly connected to Computer Vision.
## 4. Project Objectives
The project has five main objectives:
ObjectiveExplanation
Compare individual feature
types
Test classical, deep, and YOLO-based features separately
Study feature fusionCompare early fusion, late fusion, attention fusion, and YOLO
hybrid fusion
Apply dimensionality
reduction
Use PCA, autoencoder compression, and feature selection
Evaluate accuracy-efficiency
trade-off
Compare accuracy, F1-score, mAP, speed, feature size, and
model size
Identify the best practical
configuration
Recommend the most accurate and efficient representation for
waste detection/classification
## 5. Dataset
The project uses the Kaggle Trash Detection Image Dataset, which contains 4,142 imageswith
bounding box annotations in YOLO format. The original dataset includes seven classes:
Cardboard, Glass, Metal, Paper, Plastic, Trash, and Biodegradable. In the proposed project,
Biodegradable is merged with Trash, producing a final six-class taxonomy.
## Final Class List
Original ClassFinal Class
CardboardCardboard
GlassGlass
MetalMetal
PaperPaper
PlasticPlastic
TrashTrash
BiodegradableTrash
The final classification problem therefore contains six material categories.
## 6. Dataset Split
The dataset will be divided using stratified sampling:
SplitPercentageApproximate Images
## Training70%~2,900
## Validation15%~620
## Test15%~620
The split must be done at the image level, not object level. This prevents objects from the same
image appearing in both training and test sets.
Stratification is important because some classes, especially Metal, may appear less frequently
than others. The split should preserve class distribution as much as possible.
## 7. Important Methodological Clarification
This project contains two related but different tasks:
## Task A: Object Detection
Object detection means the system outputs:
bounding box + class label + confidence score
This applies mainly to:
YOLOv8-nano baseline 1.
YOLO hybrid model 2.
Detection metrics such as mAP@0.5and mAP@0.5:0.95are valid here.
## Task B: Object Crop Classification
Object crop classification means the object location is already known, and the model only
predicts the material class.
This applies to:
Classical feature classifier 1.
Deep feature classifier 2.
Early fusion models 3.
PCA fusion 4.
Autoencoder fusion 5.
Feature-selection fusion 6.
Late fusion 7.
Attention fusion 8.
These models do not perform full detection by themselves. They classify cropped object regions.
So their main metrics should be:
accuracy, macro-F1, per-class F1, feature dimension, extraction time, and classification
time
This distinction is very important. It makes the methodology technically correct.
## 8. General Pipeline
The full project pipeline will follow these stages:
## Stage 1: Dataset Preparation
Load images and YOLO annotation files. 1.
Convert YOLO labels into bounding box coordinates. 2.
Merge Biodegradable into Trash. 3.
Split dataset into train, validation, and test sets. 4.
Crop object regions from images using ground-truth boxes. 5.
Resize crops when needed, especially for EfficientNet. 6.
Stage 2: YOLO Training
A YOLOv8-nano detector will be trained on the six-class dataset.
YOLO will be used for two purposes:
As a detection baseline. 1.
As a region proposal/localization module for hybrid experiments. 2.
## Stage 3: Feature Extraction
For each object crop, three types of features will be extracted:
Classical hand-crafted features 1.
Deep EfficientNetB0 features 2.
YOLO Feature Pyramid Network features 3.
Stage 4: Fusion and Compression
Different fusion strategies will be tested:
Early feature concatenation 1.
PCA dimensionality reduction 2.
Autoencoder compression 3.
Feature selection 4.
Late probability fusion 5.
Attention-based fusion 6.
YOLO hybrid fusion 7.
## Stage 5: Evaluation
The project will evaluate both detection and classification performance.
Detection models will use mAP-based metrics.
Feature/classification models will use accuracy, macro-F1, per-class F1, time, and feature size.
## 9. Feature Extraction Design
## 9.1 Classical Feature Branch
The classical feature branch extracts interpretable visual properties from object crops.
Expected feature size: 252 dimensions
Feature TypeDimensionsPurpose
Edge features5Capture boundary and gradient information
Shape features19Describe object geometry
LBP texture features119Capture local texture patterns
GLCM texture features48Capture texture regularity and direction
HSV color histogram24Capture color distribution
Spatial pyramid features28Preserve coarse spatial layout
Lab perceptual color features7Represent human-perceived color differences
Material-specific cues2Detect highlight/metallic-like properties
Total252Full classical vector
This branch is useful because waste categories often depend on visible material properties such
as color, texture, reflectiveness, and shape.
## 9.2 Deep Hierarchical Feature Branch
The deep feature branch uses a pretrained EfficientNetB0 model.
The model is used as a feature extractor, not as a fully fine-tuned classifier.
Feature LevelDimensionsMeaning
Early-layer features256Edges, corners, colors, low-level texture
Mid-layer features512Material surfaces, local structures
Late-layer features1,280High-level semantic appearance
Total deep features2,048Full EfficientNet representation
The early, mid, and late features capture different levels of visual information.
9.3 YOLO Detection Feature Branch
YOLOv8-nano produces multiscale feature maps through its Feature Pyramid Network.
YOLO Feature MapDimensionsRole
P3256Higher-resolution features
P4256Medium-scale features
P5256Lower-resolution semantic features
Total YOLO FPN features768Detection-specific representation
These features are useful because they are directly optimized for object detection.
- Experiments to Be Conducted
The project should include eleven experimental configurations.
## Experiment Group 1: Baseline Experiments
These experiments establish the performance of each representation type before fusion.
ExperimentNameFeature SourceModelEvaluation Type
E1YOLOv8-nano
baseline
Full image YOLO
features
YOLO detectorDetection
E2Classical-only baseline252 classical featuresRandom
## Forest
## Classification
E3Deep-only baselineEfficientNet late
features
SVMClassification
E1: YOLOv8-Nano Baseline
This experiment trains YOLOv8-nano on the waste dataset.
## Purpose:
Establish the main object detection baseline.
## Outputs:
Bounding boxes •
Class labels •
Confidence scores •
## Metrics:
mAP@0.5 •
mAP@0.5:0.95 •
Per-class AP •
## FPS •
Model size •
E2: Classical-Only Baseline
This experiment extracts 252-dimensional classical features from each object crop and trains a
Random Forest classifier.
## Purpose:
Test how well interpretable visual features classify waste materials.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
E3: Deep-Only Baseline
This experiment extracts EfficientNetB0 late-layer features and trains an SVM classifier.
## Purpose:
Test how well pretrained deep semantic features classify waste objects.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
## Experiment Group 2: Early Fusion Experiments
Early fusion means features are combined first, then one classifier is trained.
## Experime
nt
NameFeatures UsedReduction MethodClassifier
E4Raw early fusionClassical + EfficientNet
early + mid + late
NoneRandom Forest
E5PCA fusionSame fused vectorPCA, 95% varianceSVM
E6Autoencoder
fusion
Same fused vector256-dimensional
bottleneck
Neural classifier
## / SVM
E7Feature-
selection fusion
Same fused vectorTop 200 featuresRandom Forest
The full early-fusion vector is:
Feature SourceDimensions
Classical features252
EfficientNet early256
EfficientNet mid512
EfficientNet late1,280
## Total2,300
## E4: Raw Early Fusion
The 2,300-dimensional fused feature vector is used directly.
## Purpose:
Test whether combining all available classical and deep features improves classification
performance.
Main question:
Does raw feature concatenation outperform individual feature branches?
E5: PCA Fusion
The 2,300-dimensional vector is compressed using PCA while retaining 95% of variance.
## Purpose:
Test whether linear dimensionality reduction can reduce feature size without large
accuracy loss.
Expected output:
Reduced feature vector, probably around 400–600 dimensions •
Classification results after compression •
## E6: Autoencoder Fusion
A shallow autoencoder compresses the 2,300-dimensional vector into a 256-dimensional
bottleneck.
Suggested structure:
## Input 2300 → Dense 512 → Dense 256 → Dense 512 → Output 2300
The 256-dimensional bottleneck is then used for classification.
## Purpose:
Test whether nonlinear compression preserves information better than PCA.
E7: Feature-Selection Fusion
A Random Forest is used to rank feature importance.
The top 200 features are selected and used for classification.
## Purpose:
Test whether only a small subset of the fused features is enough for strong classification
performance.
This experiment is useful because it may reveal which feature groups are most informative.
## Experiment Group 3: Late Fusion Experiments
Late fusion means separate classifiers are trained first, then their predictions are combined.
## Experimen
t
NameModels CombinedFusion Method
E8Average votingClassical classifier + deep
classifier
Equal probability averaging
E9Weighted votingClassical classifier + deep
classifier
## Validation-optimized
weights
## E8: Average Voting
The classical classifier and deep classifier each produce class probabilities.
The final prediction is:
average of both probability vectors
## Purpose:
Test whether simple decision-level fusion improves performance.
## E9: Weighted Voting
The classical and deep probabilities are combined using optimized weights.
## Example:
Final prediction = 0.4 classical + 0.6 deep
The best weights are selected using the validation set.
## Purpose:
Test whether one representation should be trusted more than another.
This may show that deep features are generally stronger, but classical features still add useful
material-specific information.
Experiment Group 4: Attention-Based Fusion
ExperimentNameFeatures UsedMethod
E10Attention fusionClassical + EfficientNet late featuresLearned gating network
## E10: Attention Fusion
This experiment trains a lightweight neural network that learns how much attention to give to
each feature branch.
General architecture:
Project classical features from 252 dimensions to 256 dimensions. 1.
Project EfficientNet late features from 1,280 dimensions to 256 dimensions. 2.
Concatenate both projected vectors. 3.
Learn attention weights for classical and deep features. 4.
Produce one fused 256-dimensional vector. 5.
Classify the object. 6.
## Purpose:
Test whether the system can dynamically choose which representation is more useful for
each object.
Example interpretation:
Cardboard may rely more on color and texture. •
Glass may rely more on shape, transparency, and deep visual patterns. •
Metal may rely on reflectiveness and deep features. •
This experiment is advanced and gives the project a strong research contribution.
Experiment Group 5: YOLO Hybrid Experiment
## Experimen
t
NameFeatures UsedMethodEvaluation
## E11YOLO
hybrid
YOLO FPN + classical
features
## Fusion/re-scoring
module
## Detection +
classification
E11: YOLO Hybrid
This experiment combines YOLO’s internal detection features with classical hand-crafted
features from the same detected region.
## Purpose:
Test whether YOLO detection features become stronger when combined with explicit
color, texture, and material cues.
Suggested implementation:
Run YOLO on the image. 1.
Extract predicted bounding boxes. 2.
Crop each detected object region. 3.
Extract classical features from the crop. 4.
Extract YOLO FPN features corresponding to the detected region. 5.
Concatenate YOLO FPN features with classical features. 6.
Use a classifier or re-scoring module to refine the class prediction. 7.
Final output:
YOLO bounding box •
Refined class prediction •
Refined confidence score •
Because this system still outputs bounding boxes, class labels, and confidence scores, it can be
evaluated using detection metrics.
## 11. Final Experiment Count
The project contains 11 experiments.
GroupExperimentsCount
BaselinesE1, E2, E33
Early fusionE4, E5, E6, E74
Late fusionE8, E92
Attention fusionE101
YOLO hybridE111
## Total11
So in the report, do not write “thirteen experiments.”
## Use:
eleven experimental configurations
or:
all experimental configurations
## 12. Evaluation Protocol
The evaluation should be divided into two tracks.
## Track A: Detection Evaluation
Used for:
E1: YOLOv8-nano baseline •
E11: YOLO hybrid •
MetricMeaning
mAP@0.5Detection performance at IoU threshold 0.5
mAP@0.5:0.95Stricter detection/localization metric
Per-class APDetection performance for each waste class
PrecisionCorrectness of positive detections
RecallAbility to find existing objects
FPSReal-time speed
Model footprintDeployment size
## Track B: Classification / Representation Evaluation
Used for:
## E2: Classical-only •
## E3: Deep-only •
E4: Raw early fusion •
E5: PCA fusion •
E6: Autoencoder fusion •
E7: Feature-selection fusion •
E8: Average voting •
E9: Weighted voting •
E10: Attention fusion •
MetricMeaning
AccuracyOverall classification correctness
Macro-F1Balanced performance across all classes
Per-class F1Performance for each waste material
Confusion matrixShows which classes are confused
Feature dimensionSize of the representation
Feature extraction timeCost of producing the features
Classification timeCost of predicting the class
Model sizeStorage requirement
Macro-F1 is especially important because the dataset may be imbalanced.
## 13. Controlled Variables
To ensure fair comparison, the following conditions should remain fixed:
Controlled VariableDecision
Dataset splitSame train/validation/test split for all experiments
Class mappingSame six-class taxonomy
Crop generationSame ground-truth crops for classification experiments
Image preprocessingSame resizing and normalization rules
Classifier training dataSame training objects
Test setSame test objects/images
Random seedFixed, for example 42
YOLO confidence thresholdFixed, for example 0.25
NMS IoU thresholdFixed, for example 0.5
## 14. Recommended Implementation Order
To make the project manageable, implement the experiments in this order:
Phase 1: Dataset and YOLO Baseline
Prepare dataset. 1.
Merge classes. 2.
Train YOLOv8-nano. 3.
Evaluate YOLO detection performance. 4.
## Output:
YOLO baseline results.
## Phase 2: Object Crop Dataset
Extract object crops from ground-truth bounding boxes. 1.
Save crop images and labels. 2.
Prepare train/validation/test crop-level datasets. 3.
## Output:
Clean crop classification dataset.
## Phase 3: Baseline Feature Models
Extract classical features. 1.
## Train Random Forest. 2.
Extract EfficientNet features. 3.
Train SVM. 4.
Compare classical-only and deep-only baselines. 5.
## Output:
Baseline representation results.
Phase 4: Early Fusion and Compression
Concatenate classical + EfficientNet features. 1.
Train raw fusion model. 2.
Apply PCA. 3.
Train PCA-fusion classifier. 4.
Train autoencoder. 5.
Extract bottleneck features. 6.
Train autoencoder-fusion classifier. 7.
Perform feature selection. 8.
Train selected-feature classifier. 9.
## Output:
Accuracy vs feature-dimension comparison.
## Phase 5: Late Fusion
Generate probability outputs from classical and deep classifiers. 1.
Apply average voting. 2.
Optimize voting weights on validation set. 3.
Evaluate on test set. 4.
## Output:
Decision-level fusion comparison.
## Phase 6: Attention Fusion
Build attention/gating network. 1.
Train on classical + deep features. 2.
Evaluate classification performance. 3.
Analyze attention weights by class. 4.
## Output:
Interpretable learned fusion model.
Phase 7: YOLO Hybrid
Extract YOLO FPN features. 1.
Combine YOLO FPN features with classical crop features. 2.
Train hybrid classifier/re-scoring module. 3.
Evaluate detection/classification performance. 4.
## Output:
Final hybrid detection result.
This is the most technically challenging experiment, so it should be done after the main feature-
fusion experiments are working.
## 15. Expected Results
The expected findings should be written carefully, not as guaranteed results.
## Expected Finding 1: Fusion Improves Classification
Classical features may perform well on materials with distinctive color and texture, such as
cardboard and paper.
Deep features may perform better on visually complex materials such as glass, metal, and
plastic.
Therefore, fusion is expected to outperform either feature type alone.
Expected Finding 2: PCA May Preserve Most Performance
PCA is expected to reduce the 2,300-dimensional fused vector significantly while preserving
most classification performance.
This would show that the full fused vector contains redundancy.
Expected Finding 3: Autoencoder May Help at Strong
## Compression
At very low dimensions, autoencoder compression may preserve nonlinear feature relationships
better than PCA.
This is useful if the final system is intended for deployment on limited hardware.
## Expected Finding 4: Attention Fusion May Be More
## Interpretable
The attention model may learn different branch weights for different waste classes.
For example:
ClassPossible Dominant Features
CardboardClassical color/texture
PaperClassical texture/color
GlassDeep visual patterns
MetalDeep + reflectiveness cues
PlasticMixed classical/deep
TrashMixed features
Expected Finding 5: YOLO Hybrid May Improve Class
## Confidence
The YOLO hybrid model may improve classification confidence by adding explicit material cues
to YOLO’s detection features.
However, it may not greatly improve localization because bounding box prediction still comes
mainly from YOLO.
- Main Risks and Solutions
RiskExplanationSolution
Too many experimentsEleven experiments can be time-
consuming
Implement in phases and
keep YOLO hybrid last
Confusion between
detection and classification
Classical/SVM models do not
detect objects by themselves
Use two evaluation tracks
YOLO FPN extraction may be
difficult
Accessing internal YOLO features
requires extra coding
Treat YOLO hybrid as
advanced/final experiment
Dataset imbalanceSome waste classes may have
fewer examples
Use macro-F1 and class
weights
Feature vector too large2,300 dimensions may be
computationally heavy
Use PCA, autoencoder, and
feature selection
OverfittingSome models may memorize
crop-level features
Use validation set, test set,
and fixed split
## 17. Final Deliverables
By the end of the project, the final submission should include:
DeliverableDescription
Dataset preprocessing codeClass merging, splitting, crop extraction
YOLO training resultsDetection baseline
Feature extraction scriptsClassical, EfficientNet, YOLO FPN
Trained classifiersRF, SVM, autoencoder, attention model
Experiment results tableAll 11 experiments compared
Confusion matricesFor classification experiments
Accuracy vs dimension chartShows compression trade-off
Speed comparisonFeature extraction and classification time
Final discussionBest model and explanation
Final report/presentationAcademic project documentation
## 18. Suggested Final Results Table Format
Your final report should include a table like this:
Exp.MethodFeature
## Dim.
## Accurac
y
## Macro-
## F1
mAP@0.5FPSModel
## Size
E1YOLOv8-nanoNative—————
E2Classical + RF252——N/A——
E3EfficientNet +
## SVM
## 1280——N/A——
E4Raw early fusion2300——N/A——
E5PCA fusion~400–600——N/A——
E6Autoencoder
fusion
## 256——N/A——
E7Feature selection200——N/A——
E8Average votingSeparate——N/A——
E9Weighted votingSeparate——N/A——
E10Attention fusion256——N/A——
E11YOLO hybrid768 + 252—————
Use N/Awhere mAP does not apply.
- Suggested Report Structure for Final Submission
When writing the final academic report, use this structure:
## Abstract 1.
## Introduction 2.
## Problem Statement 3.
## Research Question 4.
## Dataset Description 5.
## Preprocessing 6.
## Feature Extraction Methods 7.
## Fusion Strategies 8.
## Dimensionality Reduction Methods 9.
## Experimental Design 10.
## Evaluation Metrics 11.
## Results 12.
## Discussion 13.
## Limitations 14.
## Conclusion 15.
## References 16.
## 20. Final Feasibility Verdict
This project is viable and strong, as long as it is implemented with the correct separation
between detection and classification.
The most important correction is:
YOLO performs localization and detection. Classical, deep, and fused features mainly
evaluate object crop classification and representation quality.
The project should not claim that Random Forest, SVM, PCA, or attention fusion are full object
detectors unless they are integrated into a full detection pipeline with bounding boxes and
confidence scores.
The strongest final version is:
Train YOLOv8-nano as the detection baseline, then conduct a systematic representation
study on waste object crops using classical features, EfficientNet features, fusion methods,
and compression techniques. Finally, test whether YOLO’s detection features can be
improved through a hybrid fusion model.
## Project Description
Thursday, 14 May 20261:59 PM

Comprehensive Pre-Implementation Report
Compression-Aware Multiscale Feature Fusion for Waste
Object Detection and Classification
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
object detection and classificationacross all object sizes.
## 2. Proposed Project Title
## Recommended Title
Compression-Aware Multiscale Feature Fusion for Waste Object Detection and Classification
Alternative title:
Multiscale Feature Fusion and Dimensionality Reduction for Waste Object Detection
The first title is stronger because it clearly shows the two main technical ideas:
Feature fusion 1.
Dimensionality reduction / compression 2.
## 3. Main Research Question
The main research question is:
How can classical visual features, pretrained deep neural features, and YOLO detection
features be fused and compressed to improve waste-object classification accuracy while
reducing feature dimensionality and computational cost?
This question is clear, realistic, and directly connected to Computer Vision.
## 4. Project Objectives
The project has five main objectives:
ObjectiveExplanation
Compare individual feature
types
Test classical, deep, and YOLO-based features separately
Study feature fusionCompare early fusion, late fusion, attention fusion, and YOLO
hybrid fusion
Apply dimensionality
reduction
Use PCA, autoencoder compression, and feature selection
Evaluate accuracy-efficiency
trade-off
Compare accuracy, F1-score, mAP, speed, feature size, and
model size
Identify the best practical
configuration
Recommend the most accurate and efficient representation for
waste detection/classification
## 5. Dataset
The project uses the Kaggle Trash Detection Image Dataset, which contains 4,142 imageswith
bounding box annotations in YOLO format. The original dataset includes seven classes:
Cardboard, Glass, Metal, Paper, Plastic, Trash, and Biodegradable. In the proposed project,
Biodegradable is merged with Trash, producing a final six-class taxonomy.
## Final Class List
Original ClassFinal Class
CardboardCardboard
GlassGlass
MetalMetal
PaperPaper
PlasticPlastic
TrashTrash
BiodegradableTrash
The final classification problem therefore contains six material categories.
## 6. Dataset Split
The dataset will be divided using stratified sampling:
SplitPercentageApproximate Images
## Training70%~2,900
## Validation15%~620
## Test15%~620
The split must be done at the image level, not object level. This prevents objects from the same
image appearing in both training and test sets.
Stratification is important because some classes, especially Metal, may appear less frequently
than others. The split should preserve class distribution as much as possible.
## 7. Important Methodological Clarification
This project contains two related but different tasks:
## Task A: Object Detection
Object detection means the system outputs:
bounding box + class label + confidence score
This applies mainly to:
YOLOv8-nano baseline 1.
YOLO hybrid model 2.
Detection metrics such as mAP@0.5and mAP@0.5:0.95are valid here.
## Task B: Object Crop Classification
Object crop classification means the object location is already known, and the model only
predicts the material class.
This applies to:
Classical feature classifier 1.
Deep feature classifier 2.
Early fusion models 3.
PCA fusion 4.
Autoencoder fusion 5.
Feature-selection fusion 6.
Late fusion 7.
Attention fusion 8.
These models do not perform full detection by themselves. They classify cropped object regions.
So their main metrics should be:
accuracy, macro-F1, per-class F1, feature dimension, extraction time, and classification
time
This distinction is very important. It makes the methodology technically correct.
## 8. General Pipeline
The full project pipeline will follow these stages:
## Stage 1: Dataset Preparation
Load images and YOLO annotation files. 1.
Convert YOLO labels into bounding box coordinates. 2.
Merge Biodegradable into Trash. 3.
Split dataset into train, validation, and test sets. 4.
Crop object regions from images using ground-truth boxes. 5.
Resize crops when needed, especially for EfficientNet. 6.
Stage 2: YOLO Training
A YOLOv8-nano detector will be trained on the six-class dataset.
YOLO will be used for two purposes:
As a detection baseline. 1.
As a region proposal/localization module for hybrid experiments. 2.
## Stage 3: Feature Extraction
For each object crop, three types of features will be extracted:
Classical hand-crafted features 1.
Deep EfficientNetB0 features 2.
YOLO Feature Pyramid Network features 3.
Stage 4: Fusion and Compression
Different fusion strategies will be tested:
Early feature concatenation 1.
PCA dimensionality reduction 2.
Autoencoder compression 3.
Feature selection 4.
Late probability fusion 5.
Attention-based fusion 6.
YOLO hybrid fusion 7.
## Stage 5: Evaluation
The project will evaluate both detection and classification performance.
Detection models will use mAP-based metrics.
Feature/classification models will use accuracy, macro-F1, per-class F1, time, and feature size.
## 9. Feature Extraction Design
## 9.1 Classical Feature Branch
The classical feature branch extracts interpretable visual properties from object crops.
Expected feature size: 252 dimensions
Feature TypeDimensionsPurpose
Edge features5Capture boundary and gradient information
Shape features19Describe object geometry
LBP texture features119Capture local texture patterns
GLCM texture features48Capture texture regularity and direction
HSV color histogram24Capture color distribution
Spatial pyramid features28Preserve coarse spatial layout
Lab perceptual color features7Represent human-perceived color differences
Material-specific cues2Detect highlight/metallic-like properties
Total252Full classical vector
This branch is useful because waste categories often depend on visible material properties such
as color, texture, reflectiveness, and shape.
## 9.2 Deep Hierarchical Feature Branch
The deep feature branch uses a pretrained EfficientNetB0 model.
The model is used as a feature extractor, not as a fully fine-tuned classifier.
Feature LevelDimensionsMeaning
Early-layer features256Edges, corners, colors, low-level texture
Mid-layer features512Material surfaces, local structures
Late-layer features1,280High-level semantic appearance
Total deep features2,048Full EfficientNet representation
The early, mid, and late features capture different levels of visual information.
9.3 YOLO Detection Feature Branch
YOLOv8-nano produces multiscale feature maps through its Feature Pyramid Network.
YOLO Feature MapDimensionsRole
P3256Higher-resolution features
P4256Medium-scale features
P5256Lower-resolution semantic features
Total YOLO FPN features768Detection-specific representation
These features are useful because they are directly optimized for object detection.
- Experiments to Be Conducted
The project should include eleven experimental configurations.
## Experiment Group 1: Baseline Experiments
These experiments establish the performance of each representation type before fusion.
## Experimen
t
NameFeature SourceModelEvaluation Type
E1YOLOv8-nano
baseline
Full image YOLO
features
YOLO detectorDetection
E2Classical-only baseline252 classical featuresRandom
## Forest
## Classification
E3Deep-only baselineEfficientNet late
features
SVMClassification
E1: YOLOv8-Nano Baseline
This experiment trains YOLOv8-nano on the waste dataset.
## Purpose:
Establish the main object detection baseline.
## Outputs:
Bounding boxes •
Class labels •
Confidence scores •
## Metrics:
mAP@0.5 •
mAP@0.5:0.95 •
Per-class AP •
## FPS •
Model size •
E2: Classical-Only Baseline
This experiment extracts 252-dimensional classical features from each object crop and trains a
Random Forest classifier.
## Purpose:
Test how well interpretable visual features classify waste materials.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
E3: Deep-Only Baseline
This experiment extracts EfficientNetB0 late-layer features and trains an SVM classifier.
## Purpose:
Test how well pretrained deep semantic features classify waste objects.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
## Experiment Group 2: Early Fusion Experiments
Early fusion means features are combined first, then one classifier is trained.
## Experime
nt
NameFeatures UsedReduction MethodClassifier
E4Raw early fusionClassical + EfficientNet
early + mid + late
NoneRandom Forest
E5PCA fusionSame fused vectorPCA, 95% varianceSVM
E6Autoencoder
fusion
Same fused vector256-dimensional
bottleneck
Neural classifier
## / SVM
E7Feature-
selection fusion
Same fused vectorTop 200 featuresRandom Forest
The full early-fusion vector is:
Feature SourceDimensions
Classical features252
EfficientNet early256
EfficientNet mid512
EfficientNet late1,280
## Total2,300
## E4: Raw Early Fusion
The 2,300-dimensional fused feature vector is used directly.
## Purpose:
Test whether combining all available classical and deep features improves classification
performance.
Main question:
Does raw feature concatenation outperform individual feature branches?
E5: PCA Fusion
The 2,300-dimensional vector is compressed using PCA while retaining 95% of variance.
## Purpose:
Test whether linear dimensionality reduction can reduce feature size without large
accuracy loss.
Expected output:
Reduced feature vector, probably around 400–600 dimensions •
Classification results after compression •
## E6: Autoencoder Fusion
A shallow autoencoder compresses the 2,300-dimensional vector into a 256-dimensional
bottleneck.
Suggested structure:
## Input 2300 → Dense 512 → Dense 256 → Dense 512 → Output 2300
The 256-dimensional bottleneck is then used for classification.
## Purpose:
Test whether nonlinear compression preserves information better than PCA.
E7: Feature-Selection Fusion
A Random Forest is used to rank feature importance.
The top 200 features are selected and used for classification.
## Purpose:
Test whether only a small subset of the fused features is enough for strong classification
performance.
This experiment is useful because it may reveal which feature groups are most informative.
## Experiment Group 3: Late Fusion Experiments
Late fusion means separate classifiers are trained first, then their predictions are combined.
ExperimentNameModels CombinedFusion Method
E8Average votingClassical classifier + deep
classifier
Equal probability averaging
E9Weighted votingClassical classifier + deep
classifier
## Validation-optimized
weights
## E8: Average Voting
The classical classifier and deep classifier each produce class probabilities.
The final prediction is:
average of both probability vectors
## Purpose:
Test whether simple decision-level fusion improves performance.
## E9: Weighted Voting
The classical and deep probabilities are combined using optimized weights.
## Example:
Final prediction = 0.4 classical + 0.6 deep
The best weights are selected using the validation set.
## Purpose:
Test whether one representation should be trusted more than another.
This may show that deep features are generally stronger, but classical features still add useful
material-specific information.
Experiment Group 4: Attention-Based Fusion
ExperimentNameFeatures UsedMethod
E10Attention fusionClassical + EfficientNet late featuresLearned gating network
## E10: Attention Fusion
This experiment trains a lightweight neural network that learns how much attention to give to
each feature branch.
General architecture:
Project classical features from 252 dimensions to 256 dimensions. 1.
Project EfficientNet late features from 1,280 dimensions to 256 dimensions. 2.
Concatenate both projected vectors. 3.
Learn attention weights for classical and deep features. 4.
Produce one fused 256-dimensional vector. 5.
Classify the object. 6.
## Purpose:
Test whether the system can dynamically choose which representation is more useful for
each object.
Example interpretation:
Cardboard may rely more on color and texture. •
Glass may rely more on shape, transparency, and deep visual patterns. •
Metal may rely on reflectiveness and deep features. •
This experiment is advanced and gives the project a strong research contribution.
Experiment Group 5: YOLO Hybrid Experiment
## Experimen
t
NameFeatures UsedMethodEvaluation
## E11YOLO
hybrid
YOLO FPN + classical
features
## Fusion/re-scoring
module
## Detection +
classification
E11: YOLO Hybrid
This experiment combines YOLO’s internal detection features with classical hand-crafted
features from the same detected region.
## Purpose:
Test whether YOLO detection features become stronger when combined with explicit
color, texture, and material cues.
Suggested implementation:
Run YOLO on the image. 1.
Extract predicted bounding boxes. 2.
Crop each detected object region. 3.
Extract classical features from the crop. 4.
Extract YOLO FPN features corresponding to the detected region. 5.
Concatenate YOLO FPN features with classical features. 6.
Use a classifier or re-scoring module to refine the class prediction. 7.
Final output:
YOLO bounding box •
Refined class prediction •
Refined confidence score •
Because this system still outputs bounding boxes, class labels, and confidence scores, it can be
evaluated using detection metrics.
## 11. Final Experiment Count
The project contains 11 experiments.
GroupExperimentsCount
BaselinesE1, E2, E33
Early fusionE4, E5, E6, E74
Late fusionE8, E92
Attention fusionE101
YOLO hybridE111
## Total11
So in the report, do not write “thirteen experiments.”
## Use:
eleven experimental configurations
or:
all experimental configurations
## 12. Evaluation Protocol
The evaluation should be divided into two tracks.
## Track A: Detection Evaluation
Used for:
E1: YOLOv8-nano baseline •
E11: YOLO hybrid •
MetricMeaning
mAP@0.5Detection performance at IoU threshold 0.5
mAP@0.5:0.95Stricter detection/localization metric
Per-class APDetection performance for each waste class
PrecisionCorrectness of positive detections
RecallAbility to find existing objects
FPSReal-time speed
Model footprintDeployment size
## Track B: Classification / Representation Evaluation
Used for:
## E2: Classical-only •
## E3: Deep-only •
E4: Raw early fusion •
E5: PCA fusion •
E6: Autoencoder fusion •
E7: Feature-selection fusion •
E8: Average voting •
E9: Weighted voting •
E10: Attention fusion •
MetricMeaning
AccuracyOverall classification correctness
Macro-F1Balanced performance across all classes
Per-class F1Performance for each waste material
Confusion matrixShows which classes are confused
Feature dimensionSize of the representation
Feature extraction timeCost of producing the features
Classification timeCost of predicting the class
Model sizeStorage requirement
Macro-F1 is especially important because the dataset may be imbalanced.
## 13. Controlled Variables
To ensure fair comparison, the following conditions should remain fixed:
Controlled VariableDecision
Dataset splitSame train/validation/test split for all experiments
Class mappingSame six-class taxonomy
Crop generationSame ground-truth crops for classification experiments
Image preprocessingSame resizing and normalization rules
Classifier training dataSame training objects
Test setSame test objects/images
Random seedFixed, for example 42
YOLO confidence thresholdFixed, for example 0.25
NMS IoU thresholdFixed, for example 0.5
## 14. Recommended Implementation Order
To make the project manageable, implement the experiments in this order:
Phase 1: Dataset and YOLO Baseline
Prepare dataset. 1.
Merge classes. 2.
Train YOLOv8-nano. 3.
Evaluate YOLO detection performance. 4.
## Output:
YOLO baseline results.
## Phase 2: Object Crop Dataset
Extract object crops from ground-truth bounding boxes. 1.
Save crop images and labels. 2.
Prepare train/validation/test crop-level datasets. 3.
## Output:
Clean crop classification dataset.
## Phase 3: Baseline Feature Models
Extract classical features. 1.
## Train Random Forest. 2.
Extract EfficientNet features. 3.
Train SVM. 4.
Compare classical-only and deep-only baselines. 5.
## Output:
Baseline representation results.
Phase 4: Early Fusion and Compression
Concatenate classical + EfficientNet features. 1.
Train raw fusion model. 2.
Apply PCA. 3.
Train PCA-fusion classifier. 4.
Train autoencoder. 5.
Extract bottleneck features. 6.
Train autoencoder-fusion classifier. 7.
Perform feature selection. 8.
Train selected-feature classifier. 9.
## Output:
Accuracy vs feature-dimension comparison.
## Phase 5: Late Fusion
Generate probability outputs from classical and deep classifiers. 1.
Apply average voting. 2.
Optimize voting weights on validation set. 3.
Evaluate on test set. 4.
## Output:
Decision-level fusion comparison.
## Phase 6: Attention Fusion
Build attention/gating network. 1.
Train on classical + deep features. 2.
Evaluate classification performance. 3.
Analyze attention weights by class. 4.
## Output:
Interpretable learned fusion model.
Phase 7: YOLO Hybrid
Extract YOLO FPN features. 1.
Combine YOLO FPN features with classical crop features. 2.
Train hybrid classifier/re-scoring module. 3.
Evaluate detection/classification performance. 4.
## Output:
Final hybrid detection result.
This is the most technically challenging experiment, so it should be done after the main feature-
fusion experiments are working.
## 15. Expected Results
The expected findings should be written carefully, not as guaranteed results.
## Expected Finding 1: Fusion Improves Classification
Classical features may perform well on materials with distinctive color and texture, such as
cardboard and paper.
Deep features may perform better on visually complex materials such as glass, metal, and
plastic.
Therefore, fusion is expected to outperform either feature type alone.
Expected Finding 2: PCA May Preserve Most Performance
PCA is expected to reduce the 2,300-dimensional fused vector significantly while preserving
most classification performance.
This would show that the full fused vector contains redundancy.
Expected Finding 3: Autoencoder May Help at Strong
## Compression
At very low dimensions, autoencoder compression may preserve nonlinear feature relationships
better than PCA.
This is useful if the final system is intended for deployment on limited hardware.
## Expected Finding 4: Attention Fusion May Be More
## Interpretable
The attention model may learn different branch weights for different waste classes.
For example:
ClassPossible Dominant Features
CardboardClassical color/texture
PaperClassical texture/color
GlassDeep visual patterns
MetalDeep + reflectiveness cues
PlasticMixed classical/deep
TrashMixed features
Expected Finding 5: YOLO Hybrid May Improve Class
## Confidence
The YOLO hybrid model may improve classification confidence by adding explicit material cues
to YOLO’s detection features.
However, it may not greatly improve localization because bounding box prediction still comes
mainly from YOLO.
- Main Risks and Solutions
RiskExplanationSolution
Too many experimentsEleven experiments can be time-
consuming
Implement in phases and
keep YOLO hybrid last
Confusion between
detection and classification
Classical/SVM models do not
detect objects by themselves
Use two evaluation tracks
YOLO FPN extraction may be
difficult
Accessing internal YOLO features
requires extra coding
Treat YOLO hybrid as
advanced/final experiment
Dataset imbalanceSome waste classes may have
fewer examples
Use macro-F1 and class
weights
Feature vector too large2,300 dimensions may be
computationally heavy
Use PCA, autoencoder, and
feature selection
OverfittingSome models may memorize
crop-level features
Use validation set, test set,
and fixed split
## 17. Final Deliverables
By the end of the project, the final submission should include:
DeliverableDescription
Dataset preprocessing codeClass merging, splitting, crop extraction
YOLO training resultsDetection baseline
Feature extraction scriptsClassical, EfficientNet, YOLO FPN
Trained classifiersRF, SVM, autoencoder, attention model
Experiment results tableAll 11 experiments compared
Confusion matricesFor classification experiments
Accuracy vs dimension chartShows compression trade-off
Speed comparisonFeature extraction and classification time
Final discussionBest model and explanation
Final report/presentationAcademic project documentation
## 18. Suggested Final Results Table Format
Your final report should include a table like this:
Exp.MethodFeature
## Dim.
## Accurac
y
## Macro-
## F1
mAP@0.5FPSModel
## Size
E1YOLOv8-nanoNative—————
E2Classical + RF252——N/A——
E3EfficientNet +
## SVM
## 1280——N/A——
E4Raw early fusion2300——N/A——
E5PCA fusion~400–600——N/A——
E6Autoencoder
fusion
## 256——N/A——
E7Feature selection200——N/A——
E8Average votingSeparate——N/A——
E9Weighted votingSeparate——N/A——
E10Attention fusion256——N/A——
E11YOLO hybrid768 + 252—————
Use N/Awhere mAP does not apply.
- Suggested Report Structure for Final Submission
When writing the final academic report, use this structure:
## Abstract 1.
## Introduction 2.
## Problem Statement 3.
## Research Question 4.
## Dataset Description 5.
## Preprocessing 6.
## Feature Extraction Methods 7.
## Fusion Strategies 8.
## Dimensionality Reduction Methods 9.
## Experimental Design 10.
## Evaluation Metrics 11.
## Results 12.
## Discussion 13.
## Limitations 14.
## Conclusion 15.
## References 16.
## 20. Final Feasibility Verdict
This project is viable and strong, as long as it is implemented with the correct separation
between detection and classification.
The most important correction is:
YOLO performs localization and detection. Classical, deep, and fused features mainly
evaluate object crop classification and representation quality.
The project should not claim that Random Forest, SVM, PCA, or attention fusion are full object
detectors unless they are integrated into a full detection pipeline with bounding boxes and
confidence scores.
The strongest final version is:
Train YOLOv8-nano as the detection baseline, then conduct a systematic representation
study on waste object crops using classical features, EfficientNet features, fusion methods,
and compression techniques. Finally, test whether YOLO’s detection features can be
improved through a hybrid fusion model.
## Project Description
Thursday, 14 May 20261:59 PM

Comprehensive Pre-Implementation Report
Compression-Aware Multiscale Feature Fusion for Waste
Object Detection and Classification
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
object detection and classificationacross all object sizes.
## 2. Proposed Project Title
## Recommended Title
Compression-Aware Multiscale Feature Fusion for Waste Object Detection and Classification
Alternative title:
Multiscale Feature Fusion and Dimensionality Reduction for Waste Object Detection
The first title is stronger because it clearly shows the two main technical ideas:
Feature fusion 1.
Dimensionality reduction / compression 2.
## 3. Main Research Question
The main research question is:
How can classical visual features, pretrained deep neural features, and YOLO detection
features be fused and compressed to improve waste-object classification accuracy while
reducing feature dimensionality and computational cost?
This question is clear, realistic, and directly connected to Computer Vision.
## 4. Project Objectives
The project has five main objectives:
ObjectiveExplanation
Compare individual feature
types
Test classical, deep, and YOLO-based features separately
Study feature fusionCompare early fusion, late fusion, attention fusion, and YOLO
hybrid fusion
Apply dimensionality
reduction
Use PCA, autoencoder compression, and feature selection
Evaluate accuracy-efficiency
trade-off
Compare accuracy, F1-score, mAP, speed, feature size, and
model size
Identify the best practical
configuration
Recommend the most accurate and efficient representation for
waste detection/classification
## 5. Dataset
The project uses the Kaggle Trash Detection Image Dataset, which contains 4,142 imageswith
bounding box annotations in YOLO format. The original dataset includes seven classes:
Cardboard, Glass, Metal, Paper, Plastic, Trash, and Biodegradable. In the proposed project,
Biodegradable is merged with Trash, producing a final six-class taxonomy.
## Final Class List
Original ClassFinal Class
CardboardCardboard
GlassGlass
MetalMetal
PaperPaper
PlasticPlastic
TrashTrash
BiodegradableTrash
The final classification problem therefore contains six material categories.
## 6. Dataset Split
The dataset will be divided using stratified sampling:
SplitPercentageApproximate Images
## Training70%~2,900
## Validation15%~620
## Test15%~620
The split must be done at the image level, not object level. This prevents objects from the same
image appearing in both training and test sets.
Stratification is important because some classes, especially Metal, may appear less frequently
than others. The split should preserve class distribution as much as possible.
## 7. Important Methodological Clarification
This project contains two related but different tasks:
## Task A: Object Detection
Object detection means the system outputs:
bounding box + class label + confidence score
This applies mainly to:
YOLOv8-nano baseline 1.
YOLO hybrid model 2.
Detection metrics such as mAP@0.5and mAP@0.5:0.95are valid here.
## Task B: Object Crop Classification
Object crop classification means the object location is already known, and the model only
predicts the material class.
This applies to:
Classical feature classifier 1.
Deep feature classifier 2.
Early fusion models 3.
PCA fusion 4.
Autoencoder fusion 5.
Feature-selection fusion 6.
Late fusion 7.
Attention fusion 8.
These models do not perform full detection by themselves. They classify cropped object regions.
So their main metrics should be:
accuracy, macro-F1, per-class F1, feature dimension, extraction time, and classification
time
This distinction is very important. It makes the methodology technically correct.
## 8. General Pipeline
The full project pipeline will follow these stages:
## Stage 1: Dataset Preparation
Load images and YOLO annotation files. 1.
Convert YOLO labels into bounding box coordinates. 2.
Merge Biodegradable into Trash. 3.
Split dataset into train, validation, and test sets. 4.
Crop object regions from images using ground-truth boxes. 5.
Resize crops when needed, especially for EfficientNet. 6.
Stage 2: YOLO Training
A YOLOv8-nano detector will be trained on the six-class dataset.
YOLO will be used for two purposes:
As a detection baseline. 1.
As a region proposal/localization module for hybrid experiments. 2.
## Stage 3: Feature Extraction
For each object crop, three types of features will be extracted:
Classical hand-crafted features 1.
Deep EfficientNetB0 features 2.
YOLO Feature Pyramid Network features 3.
Stage 4: Fusion and Compression
Different fusion strategies will be tested:
Early feature concatenation 1.
PCA dimensionality reduction 2.
Autoencoder compression 3.
Feature selection 4.
Late probability fusion 5.
Attention-based fusion 6.
YOLO hybrid fusion 7.
## Stage 5: Evaluation
The project will evaluate both detection and classification performance.
Detection models will use mAP-based metrics.
Feature/classification models will use accuracy, macro-F1, per-class F1, time, and feature size.
## 9. Feature Extraction Design
## 9.1 Classical Feature Branch
The classical feature branch extracts interpretable visual properties from object crops.
Expected feature size: 252 dimensions
Feature TypeDimensionsPurpose
Edge features5Capture boundary and gradient information
Shape features19Describe object geometry
LBP texture features119Capture local texture patterns
GLCM texture features48Capture texture regularity and direction
HSV color histogram24Capture color distribution
Spatial pyramid features28Preserve coarse spatial layout
Lab perceptual color features7Represent human-perceived color differences
Material-specific cues2Detect highlight/metallic-like properties
Total252Full classical vector
This branch is useful because waste categories often depend on visible material properties such
as color, texture, reflectiveness, and shape.
## 9.2 Deep Hierarchical Feature Branch
The deep feature branch uses a pretrained EfficientNetB0 model.
The model is used as a feature extractor, not as a fully fine-tuned classifier.
Feature LevelDimensionsMeaning
Early-layer features256Edges, corners, colors, low-level texture
Mid-layer features512Material surfaces, local structures
Late-layer features1,280High-level semantic appearance
Total deep features2,048Full EfficientNet representation
The early, mid, and late features capture different levels of visual information.
9.3 YOLO Detection Feature Branch
YOLOv8-nano produces multiscale feature maps through its Feature Pyramid Network.
YOLO Feature MapDimensionsRole
P3256Higher-resolution features
P4256Medium-scale features
P5256Lower-resolution semantic features
Total YOLO FPN features768Detection-specific representation
These features are useful because they are directly optimized for object detection.
- Experiments to Be Conducted
The project should include eleven experimental configurations.
## Experiment Group 1: Baseline Experiments
These experiments establish the performance of each representation type before fusion.
ExperimentNameFeature SourceModelEvaluation Type
E1YOLOv8-nano
baseline
Full image YOLO
features
YOLO detectorDetection
E2Classical-only baseline252 classical featuresRandom
## Forest
## Classification
E3Deep-only baselineEfficientNet late
features
SVMClassification
E1: YOLOv8-Nano Baseline
This experiment trains YOLOv8-nano on the waste dataset.
## Purpose:
Establish the main object detection baseline.
## Outputs:
Bounding boxes •
Class labels •
Confidence scores •
## Metrics:
mAP@0.5 •
mAP@0.5:0.95 •
Per-class AP •
## FPS •
Model size •
E2: Classical-Only Baseline
This experiment extracts 252-dimensional classical features from each object crop and trains a
Random Forest classifier.
## Purpose:
Test how well interpretable visual features classify waste materials.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
E3: Deep-Only Baseline
This experiment extracts EfficientNetB0 late-layer features and trains an SVM classifier.
## Purpose:
Test how well pretrained deep semantic features classify waste objects.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
## Experiment Group 2: Early Fusion Experiments
Early fusion means features are combined first, then one classifier is trained.
## Experime
nt
NameFeatures UsedReduction MethodClassifier
E4Raw early fusionClassical + EfficientNet
early + mid + late
NoneRandom Forest
E5PCA fusionSame fused vectorPCA, 95% varianceSVM
E6Autoencoder
fusion
Same fused vector256-dimensional
bottleneck
Neural classifier
## / SVM
E7Feature-
selection fusion
Same fused vectorTop 200 featuresRandom Forest
The full early-fusion vector is:
Feature SourceDimensions
Classical features252
EfficientNet early256
EfficientNet mid512
EfficientNet late1,280
## Total2,300
## E4: Raw Early Fusion
The 2,300-dimensional fused feature vector is used directly.
## Purpose:
Test whether combining all available classical and deep features improves classification
performance.
Main question:
Does raw feature concatenation outperform individual feature branches?
E5: PCA Fusion
The 2,300-dimensional vector is compressed using PCA while retaining 95% of variance.
## Purpose:
Test whether linear dimensionality reduction can reduce feature size without large
accuracy loss.
Expected output:
Reduced feature vector, probably around 400–600 dimensions •
Classification results after compression •
## E6: Autoencoder Fusion
A shallow autoencoder compresses the 2,300-dimensional vector into a 256-dimensional
bottleneck.
Suggested structure:
## Input 2300 → Dense 512 → Dense 256 → Dense 512 → Output 2300
The 256-dimensional bottleneck is then used for classification.
## Purpose:
Test whether nonlinear compression preserves information better than PCA.
E7: Feature-Selection Fusion
A Random Forest is used to rank feature importance.
The top 200 features are selected and used for classification.
## Purpose:
Test whether only a small subset of the fused features is enough for strong classification
performance.
This experiment is useful because it may reveal which feature groups are most informative.
## Experiment Group 3: Late Fusion Experiments
Late fusion means separate classifiers are trained first, then their predictions are combined.
ExperimentNameModels CombinedFusion Method
E8Average votingClassical classifier + deep
classifier
Equal probability averaging
E9Weighted votingClassical classifier + deep
classifier
## Validation-optimized
weights
## E8: Average Voting
The classical classifier and deep classifier each produce class probabilities.
The final prediction is:
average of both probability vectors
## Purpose:
Test whether simple decision-level fusion improves performance.
## E9: Weighted Voting
The classical and deep probabilities are combined using optimized weights.
## Example:
Final prediction = 0.4 classical + 0.6 deep
The best weights are selected using the validation set.
## Purpose:
Test whether one representation should be trusted more than another.
This may show that deep features are generally stronger, but classical features still add useful
material-specific information.
Experiment Group 4: Attention-Based Fusion
ExperimentNameFeatures UsedMethod
E10Attention fusionClassical + EfficientNet late featuresLearned gating network
## E10: Attention Fusion
This experiment trains a lightweight neural network that learns how much attention to give to
each feature branch.
General architecture:
Project classical features from 252 dimensions to 256 dimensions. 1.
Project EfficientNet late features from 1,280 dimensions to 256 dimensions. 2.
Concatenate both projected vectors. 3.
Learn attention weights for classical and deep features. 4.
Produce one fused 256-dimensional vector. 5.
Classify the object. 6.
## Purpose:
Test whether the system can dynamically choose which representation is more useful for
each object.
Example interpretation:
Cardboard may rely more on color and texture. •
Glass may rely more on shape, transparency, and deep visual patterns. •
Metal may rely on reflectiveness and deep features. •
This experiment is advanced and gives the project a strong research contribution.
Experiment Group 5: YOLO Hybrid Experiment
## Experimen
t
NameFeatures UsedMethodEvaluation
## E11YOLO
hybrid
YOLO FPN + classical
features
## Fusion/re-scoring
module
## Detection +
classification
E11: YOLO Hybrid
This experiment combines YOLO’s internal detection features with classical hand-crafted
features from the same detected region.
## Purpose:
Test whether YOLO detection features become stronger when combined with explicit
color, texture, and material cues.
Suggested implementation:
Run YOLO on the image. 1.
Extract predicted bounding boxes. 2.
Crop each detected object region. 3.
Extract classical features from the crop. 4.
Extract YOLO FPN features corresponding to the detected region. 5.
Concatenate YOLO FPN features with classical features. 6.
Use a classifier or re-scoring module to refine the class prediction. 7.
Final output:
YOLO bounding box •
Refined class prediction •
Refined confidence score •
Because this system still outputs bounding boxes, class labels, and confidence scores, it can be
evaluated using detection metrics.
## 11. Final Experiment Count
The project contains 11 experiments.
GroupExperimentsCount
BaselinesE1, E2, E33
Early fusionE4, E5, E6, E74
Late fusionE8, E92
Attention fusionE101
YOLO hybridE111
## Total11
So in the report, do not write “thirteen experiments.”
## Use:
eleven experimental configurations
or:
all experimental configurations
## 12. Evaluation Protocol
The evaluation should be divided into two tracks.
## Track A: Detection Evaluation
Used for:
E1: YOLOv8-nano baseline •
E11: YOLO hybrid •
MetricMeaning
mAP@0.5Detection performance at IoU threshold 0.5
mAP@0.5:0.95Stricter detection/localization metric
Per-class APDetection performance for each waste class
PrecisionCorrectness of positive detections
RecallAbility to find existing objects
FPSReal-time speed
Model footprintDeployment size
## Track B: Classification / Representation Evaluation
Used for:
## E2: Classical-only •
## E3: Deep-only •
E4: Raw early fusion •
E5: PCA fusion •
E6: Autoencoder fusion •
E7: Feature-selection fusion •
E8: Average voting •
E9: Weighted voting •
E10: Attention fusion •
MetricMeaning
AccuracyOverall classification correctness
Macro-F1Balanced performance across all classes
Per-class F1Performance for each waste material
Confusion matrixShows which classes are confused
Feature dimensionSize of the representation
Feature extraction timeCost of producing the features
Classification timeCost of predicting the class
Model sizeStorage requirement
Macro-F1 is especially important because the dataset may be imbalanced.
## 13. Controlled Variables
To ensure fair comparison, the following conditions should remain fixed:
Controlled VariableDecision
Dataset splitSame train/validation/test split for all experiments
Class mappingSame six-class taxonomy
Crop generationSame ground-truth crops for classification experiments
Image preprocessingSame resizing and normalization rules
Classifier training dataSame training objects
Test setSame test objects/images
Random seedFixed, for example 42
YOLO confidence thresholdFixed, for example 0.25
NMS IoU thresholdFixed, for example 0.5
## 14. Recommended Implementation Order
To make the project manageable, implement the experiments in this order:
Phase 1: Dataset and YOLO Baseline
Prepare dataset. 1.
Merge classes. 2.
Train YOLOv8-nano. 3.
Evaluate YOLO detection performance. 4.
## Output:
YOLO baseline results.
## Phase 2: Object Crop Dataset
Extract object crops from ground-truth bounding boxes. 1.
Save crop images and labels. 2.
Prepare train/validation/test crop-level datasets. 3.
## Output:
Clean crop classification dataset.
## Phase 3: Baseline Feature Models
Extract classical features. 1.
## Train Random Forest. 2.
Extract EfficientNet features. 3.
Train SVM. 4.
Compare classical-only and deep-only baselines. 5.
## Output:
Baseline representation results.
Phase 4: Early Fusion and Compression
Concatenate classical + EfficientNet features. 1.
Train raw fusion model. 2.
Apply PCA. 3.
Train PCA-fusion classifier. 4.
Train autoencoder. 5.
Extract bottleneck features. 6.
Train autoencoder-fusion classifier. 7.
Perform feature selection. 8.
Train selected-feature classifier. 9.
## Output:
Accuracy vs feature-dimension comparison.
## Phase 5: Late Fusion
Generate probability outputs from classical and deep classifiers. 1.
Apply average voting. 2.
Optimize voting weights on validation set. 3.
Evaluate on test set. 4.
## Output:
Decision-level fusion comparison.
## Phase 6: Attention Fusion
Build attention/gating network. 1.
Train on classical + deep features. 2.
Evaluate classification performance. 3.
Analyze attention weights by class. 4.
## Output:
Interpretable learned fusion model.
Phase 7: YOLO Hybrid
Extract YOLO FPN features. 1.
Combine YOLO FPN features with classical crop features. 2.
Train hybrid classifier/re-scoring module. 3.
Evaluate detection/classification performance. 4.
## Output:
Final hybrid detection result.
This is the most technically challenging experiment, so it should be done after the main feature-
fusion experiments are working.
## 15. Expected Results
The expected findings should be written carefully, not as guaranteed results.
## Expected Finding 1: Fusion Improves Classification
Classical features may perform well on materials with distinctive color and texture, such as
cardboard and paper.
Deep features may perform better on visually complex materials such as glass, metal, and
plastic.
Therefore, fusion is expected to outperform either feature type alone.
Expected Finding 2: PCA May Preserve Most Performance
PCA is expected to reduce the 2,300-dimensional fused vector significantly while preserving
most classification performance.
This would show that the full fused vector contains redundancy.
Expected Finding 3: Autoencoder May Help at Strong
## Compression
At very low dimensions, autoencoder compression may preserve nonlinear feature relationships
better than PCA.
This is useful if the final system is intended for deployment on limited hardware.
## Expected Finding 4: Attention Fusion May Be More
## Interpretable
The attention model may learn different branch weights for different waste classes.
For example:
ClassPossible Dominant Features
CardboardClassical color/texture
PaperClassical texture/color
GlassDeep visual patterns
MetalDeep + reflectiveness cues
PlasticMixed classical/deep
TrashMixed features
Expected Finding 5: YOLO Hybrid May Improve Class
## Confidence
The YOLO hybrid model may improve classification confidence by adding explicit material cues
to YOLO’s detection features.
However, it may not greatly improve localization because bounding box prediction still comes
mainly from YOLO.
- Main Risks and Solutions
RiskExplanationSolution
Too many experimentsEleven experiments can be time-
consuming
Implement in phases and
keep YOLO hybrid last
Confusion between
detection and classification
Classical/SVM models do not
detect objects by themselves
Use two evaluation tracks
YOLO FPN extraction may be
difficult
Accessing internal YOLO features
requires extra coding
Treat YOLO hybrid as
advanced/final experiment
Dataset imbalanceSome waste classes may have
fewer examples
Use macro-F1 and class
weights
Feature vector too large2,300 dimensions may be
computationally heavy
Use PCA, autoencoder, and
feature selection
OverfittingSome models may memorize
crop-level features
Use validation set, test set,
and fixed split
## 17. Final Deliverables
By the end of the project, the final submission should include:
DeliverableDescription
Dataset preprocessing codeClass merging, splitting, crop extraction
YOLO training resultsDetection baseline
Feature extraction scriptsClassical, EfficientNet, YOLO FPN
Trained classifiersRF, SVM, autoencoder, attention model
Experiment results tableAll 11 experiments compared
Confusion matricesFor classification experiments
Accuracy vs dimension chartShows compression trade-off
Speed comparisonFeature extraction and classification time
Final discussionBest model and explanation
Final report/presentationAcademic project documentation
## 18. Suggested Final Results Table Format
Your final report should include a table like this:
Exp.MethodFeature
## Dim.
## Accurac
y
## Macro-
## F1
mAP@0.5FPSModel
## Size
E1YOLOv8-nanoNative—————
E2Classical + RF252——N/A——
E3EfficientNet +
## SVM
## 1280——N/A——
E4Raw early fusion2300——N/A——
E5PCA fusion~400–600——N/A——
E6Autoencoder
fusion
## 256——N/A——
E7Feature selection200——N/A——
E8Average votingSeparate——N/A——
E9Weighted votingSeparate——N/A——
E10Attention fusion256——N/A——
E11YOLO hybrid768 + 252—————
Use N/Awhere mAP does not apply.
- Suggested Report Structure for Final Submission
When writing the final academic report, use this structure:
## Abstract 1.
## Introduction 2.
## Problem Statement 3.
## Research Question 4.
## Dataset Description 5.
## Preprocessing 6.
## Feature Extraction Methods 7.
## Fusion Strategies 8.
## Dimensionality Reduction Methods 9.
## Experimental Design 10.
## Evaluation Metrics 11.
## Results 12.
## Discussion 13.
## Limitations 14.
## Conclusion 15.
## References 16.
## 20. Final Feasibility Verdict
This project is viable and strong, as long as it is implemented with the correct separation
between detection and classification.
The most important correction is:
YOLO performs localization and detection. Classical, deep, and fused features mainly
evaluate object crop classification and representation quality.
The project should not claim that Random Forest, SVM, PCA, or attention fusion are full object
detectors unless they are integrated into a full detection pipeline with bounding boxes and
confidence scores.
The strongest final version is:
Train YOLOv8-nano as the detection baseline, then conduct a systematic representation
study on waste object crops using classical features, EfficientNet features, fusion methods,
and compression techniques. Finally, test whether YOLO’s detection features can be
improved through a hybrid fusion model.
## Project Description
Thursday, 14 May 20261:59 PM

Comprehensive Pre-Implementation Report
Compression-Aware Multiscale Feature Fusion for Waste
Object Detection and Classification
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
object detection and classificationacross all object sizes.
## 2. Proposed Project Title
## Recommended Title
Compression-Aware Multiscale Feature Fusion for Waste Object Detection and Classification
Alternative title:
Multiscale Feature Fusion and Dimensionality Reduction for Waste Object Detection
The first title is stronger because it clearly shows the two main technical ideas:
Feature fusion 1.
Dimensionality reduction / compression 2.
## 3. Main Research Question
The main research question is:
How can classical visual features, pretrained deep neural features, and YOLO detection
features be fused and compressed to improve waste-object classification accuracy while
reducing feature dimensionality and computational cost?
This question is clear, realistic, and directly connected to Computer Vision.
## 4. Project Objectives
The project has five main objectives:
ObjectiveExplanation
Compare individual feature
types
Test classical, deep, and YOLO-based features separately
Study feature fusionCompare early fusion, late fusion, attention fusion, and YOLO
hybrid fusion
Apply dimensionality
reduction
Use PCA, autoencoder compression, and feature selection
Evaluate accuracy-efficiency
trade-off
Compare accuracy, F1-score, mAP, speed, feature size, and
model size
Identify the best practical
configuration
Recommend the most accurate and efficient representation for
waste detection/classification
## 5. Dataset
The project uses the Kaggle Trash Detection Image Dataset, which contains 4,142 imageswith
bounding box annotations in YOLO format. The original dataset includes seven classes:
Cardboard, Glass, Metal, Paper, Plastic, Trash, and Biodegradable. In the proposed project,
Biodegradable is merged with Trash, producing a final six-class taxonomy.
## Final Class List
Original ClassFinal Class
CardboardCardboard
GlassGlass
MetalMetal
PaperPaper
PlasticPlastic
TrashTrash
BiodegradableTrash
The final classification problem therefore contains six material categories.
## 6. Dataset Split
The dataset will be divided using stratified sampling:
SplitPercentageApproximate Images
## Training70%~2,900
## Validation15%~620
## Test15%~620
The split must be done at the image level, not object level. This prevents objects from the same
image appearing in both training and test sets.
Stratification is important because some classes, especially Metal, may appear less frequently
than others. The split should preserve class distribution as much as possible.
## 7. Important Methodological Clarification
This project contains two related but different tasks:
## Task A: Object Detection
Object detection means the system outputs:
bounding box + class label + confidence score
This applies mainly to:
YOLOv8-nano baseline 1.
YOLO hybrid model 2.
Detection metrics such as mAP@0.5and mAP@0.5:0.95are valid here.
## Task B: Object Crop Classification
Object crop classification means the object location is already known, and the model only
predicts the material class.
This applies to:
Classical feature classifier 1.
Deep feature classifier 2.
Early fusion models 3.
PCA fusion 4.
Autoencoder fusion 5.
Feature-selection fusion 6.
Late fusion 7.
Attention fusion 8.
These models do not perform full detection by themselves. They classify cropped object regions.
So their main metrics should be:
accuracy, macro-F1, per-class F1, feature dimension, extraction time, and classification
time
This distinction is very important. It makes the methodology technically correct.
## 8. General Pipeline
The full project pipeline will follow these stages:
## Stage 1: Dataset Preparation
Load images and YOLO annotation files. 1.
Convert YOLO labels into bounding box coordinates. 2.
Merge Biodegradable into Trash. 3.
Split dataset into train, validation, and test sets. 4.
Crop object regions from images using ground-truth boxes. 5.
Resize crops when needed, especially for EfficientNet. 6.
Stage 2: YOLO Training
A YOLOv8-nano detector will be trained on the six-class dataset.
YOLO will be used for two purposes:
As a detection baseline. 1.
As a region proposal/localization module for hybrid experiments. 2.
## Stage 3: Feature Extraction
For each object crop, three types of features will be extracted:
Classical hand-crafted features 1.
Deep EfficientNetB0 features 2.
YOLO Feature Pyramid Network features 3.
Stage 4: Fusion and Compression
Different fusion strategies will be tested:
Early feature concatenation 1.
PCA dimensionality reduction 2.
Autoencoder compression 3.
Feature selection 4.
Late probability fusion 5.
Attention-based fusion 6.
YOLO hybrid fusion 7.
## Stage 5: Evaluation
The project will evaluate both detection and classification performance.
Detection models will use mAP-based metrics.
Feature/classification models will use accuracy, macro-F1, per-class F1, time, and feature size.
## 9. Feature Extraction Design
## 9.1 Classical Feature Branch
The classical feature branch extracts interpretable visual properties from object crops.
Expected feature size: 252 dimensions
Feature TypeDimensionsPurpose
Edge features5Capture boundary and gradient information
Shape features19Describe object geometry
LBP texture features119Capture local texture patterns
GLCM texture features48Capture texture regularity and direction
HSV color histogram24Capture color distribution
Spatial pyramid features28Preserve coarse spatial layout
Lab perceptual color features7Represent human-perceived color differences
Material-specific cues2Detect highlight/metallic-like properties
Total252Full classical vector
This branch is useful because waste categories often depend on visible material properties such
as color, texture, reflectiveness, and shape.
## 9.2 Deep Hierarchical Feature Branch
The deep feature branch uses a pretrained EfficientNetB0 model.
The model is used as a feature extractor, not as a fully fine-tuned classifier.
Feature LevelDimensionsMeaning
Early-layer features256Edges, corners, colors, low-level texture
Mid-layer features512Material surfaces, local structures
Late-layer features1,280High-level semantic appearance
Total deep features2,048Full EfficientNet representation
The early, mid, and late features capture different levels of visual information.
9.3 YOLO Detection Feature Branch
YOLOv8-nano produces multiscale feature maps through its Feature Pyramid Network.
YOLO Feature MapDimensionsRole
P3256Higher-resolution features
P4256Medium-scale features
P5256Lower-resolution semantic features
Total YOLO FPN features768Detection-specific representation
These features are useful because they are directly optimized for object detection.
- Experiments to Be Conducted
The project should include eleven experimental configurations.
## Experiment Group 1: Baseline Experiments
These experiments establish the performance of each representation type before fusion.
ExperimentNameFeature SourceModelEvaluation Type
E1YOLOv8-nano
baseline
Full image YOLO
features
YOLO detectorDetection
E2Classical-only baseline252 classical featuresRandom
## Forest
## Classification
E3Deep-only baselineEfficientNet late
features
SVMClassification
E1: YOLOv8-Nano Baseline
This experiment trains YOLOv8-nano on the waste dataset.
## Purpose:
Establish the main object detection baseline.
## Outputs:
Bounding boxes •
Class labels •
Confidence scores •
## Metrics:
mAP@0.5 •
mAP@0.5:0.95 •
Per-class AP •
## FPS •
Model size •
E2: Classical-Only Baseline
This experiment extracts 252-dimensional classical features from each object crop and trains a
Random Forest classifier.
## Purpose:
Test how well interpretable visual features classify waste materials.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
E3: Deep-Only Baseline
This experiment extracts EfficientNetB0 late-layer features and trains an SVM classifier.
## Purpose:
Test how well pretrained deep semantic features classify waste objects.
## Metrics:
## Accuracy •
Macro-F1 •
## Per-class F1 •
Feature extraction time •
Classification time •
## Experiment Group 2: Early Fusion Experiments
Early fusion means features are combined first, then one classifier is trained.
## Experime
nt
NameFeatures UsedReduction MethodClassifier
E4Raw early fusionClassical + EfficientNet
early + mid + late
NoneRandom Forest
E5PCA fusionSame fused vectorPCA, 95% varianceSVM
E6Autoencoder
fusion
Same fused vector256-dimensional
bottleneck
Neural classifier
## / SVM
E7Feature-
selection fusion
Same fused vectorTop 200 featuresRandom Forest
The full early-fusion vector is:
Feature SourceDimensions
Classical features252
EfficientNet early256
EfficientNet mid512
EfficientNet late1,280
## Total2,300
## E4: Raw Early Fusion
The 2,300-dimensional fused feature vector is used directly.
## Purpose:
Test whether combining all available classical and deep features improves classification
performance.
Main question:
Does raw feature concatenation outperform individual feature branches?
E5: PCA Fusion
The 2,300-dimensional vector is compressed using PCA while retaining 95% of variance.
## Purpose:
Test whether linear dimensionality reduction can reduce feature size without large
accuracy loss.
Expected output:
Reduced feature vector, probably around 400–600 dimensions •
Classification results after compression •
## E6: Autoencoder Fusion
A shallow autoencoder compresses the 2,300-dimensional vector into a 256-dimensional
bottleneck.
Suggested structure:
## Input 2300 → Dense 512 → Dense 256 → Dense 512 → Output 2300
The 256-dimensional bottleneck is then used for classification.
## Purpose:
Test whether nonlinear compression preserves information better than PCA.
E7: Feature-Selection Fusion
A Random Forest is used to rank feature importance.
The top 200 features are selected and used for classification.
## Purpose:
Test whether only a small subset of the fused features is enough for strong classification
performance.
This experiment is useful because it may reveal which feature groups are most informative.
## Experiment Group 3: Late Fusion Experiments
Late fusion means separate classifiers are trained first, then their predictions are combined.
ExperimentNameModels CombinedFusion Method
E8Average votingClassical classifier + deep
classifier
Equal probability averaging
E9Weighted
voting
Classical classifier + deep
classifier
## Validation-optimized
weights
## E8: Average Voting
The classical classifier and deep classifier each produce class probabilities.
The final prediction is:
average of both probability vectors
## Purpose:
Test whether simple decision-level fusion improves performance.
## E9: Weighted Voting
The classical and deep probabilities are combined using optimized weights.
## Example:
Final prediction = 0.4 classical + 0.6 deep
The best weights are selected using the validation set.
## Purpose:
Test whether one representation should be trusted more than another.
This may show that deep features are generally stronger, but classical features still add useful
material-specific information.
Experiment Group 4: Attention-Based Fusion
ExperimentNameFeatures UsedMethod
E10Attention fusionClassical + EfficientNet late featuresLearned gating network
## E10: Attention Fusion
This experiment trains a lightweight neural network that learns how much attention to give to
each feature branch.
General architecture:
Project classical features from 252 dimensions to 256 dimensions. 1.
Project EfficientNet late features from 1,280 dimensions to 256 dimensions. 2.
Concatenate both projected vectors. 3.
Learn attention weights for classical and deep features. 4.
Produce one fused 256-dimensional vector. 5.
Classify the object. 6.
## Purpose:
Test whether the system can dynamically choose which representation is more useful for
each object.
Example interpretation:
Cardboard may rely more on color and texture. •
Glass may rely more on shape, transparency, and deep visual patterns. •
Metal may rely on reflectiveness and deep features. •
This experiment is advanced and gives the project a strong research contribution.
Experiment Group 5: YOLO Hybrid Experiment
## Experimen
t
NameFeatures UsedMethodEvaluation
## E11YOLO
hybrid
YOLO FPN + classical
features
## Fusion/re-scoring
module
## Detection +
classification
E11: YOLO Hybrid
This experiment combines YOLO’s internal detection features with classical hand-crafted
features from the same detected region.
## Purpose:
Test whether YOLO detection features become stronger when combined with explicit
color, texture, and material cues.
Suggested implementation:
Run YOLO on the image. 1.
Extract predicted bounding boxes. 2.
Crop each detected object region. 3.
Extract classical features from the crop. 4.
Extract YOLO FPN features corresponding to the detected region. 5.
Concatenate YOLO FPN features with classical features. 6.
Use a classifier or re-scoring module to refine the class prediction. 7.
Final output:
YOLO bounding box •
Refined class prediction •
Refined confidence score •
Because this system still outputs bounding boxes, class labels, and confidence scores, it can be
evaluated using detection metrics.
## 11. Final Experiment Count
The project contains 11 experiments.
GroupExperimentsCount
BaselinesE1, E2, E33
Early fusionE4, E5, E6, E74
Late fusionE8, E92
Attention fusionE101
YOLO hybridE111
## Total11
So in the report, do not write “thirteen experiments.”
## Use:
eleven experimental configurations
or:
all experimental configurations
## 12. Evaluation Protocol
The evaluation should be divided into two tracks.
## Track A: Detection Evaluation
Used for:
E1: YOLOv8-nano baseline •
E11: YOLO hybrid •
MetricMeaning
mAP@0.5Detection performance at IoU threshold 0.5
mAP@0.5:0.95Stricter detection/localization metric
Per-class APDetection performance for each waste class
PrecisionCorrectness of positive detections
RecallAbility to find existing objects
FPSReal-time speed
Model footprintDeployment size
## Track B: Classification / Representation Evaluation
Used for:
## E2: Classical-only •
## E3: Deep-only •
E4: Raw early fusion •
E5: PCA fusion •
E6: Autoencoder fusion •
E7: Feature-selection fusion •
E8: Average voting •
E9: Weighted voting •
E10: Attention fusion •
MetricMeaning
AccuracyOverall classification correctness
Macro-F1Balanced performance across all classes
Per-class F1Performance for each waste material
Confusion matrixShows which classes are confused
Feature dimensionSize of the representation
Feature extraction timeCost of producing the features
Classification timeCost of predicting the class
Model sizeStorage requirement
Macro-F1 is especially important because the dataset may be imbalanced.
## 13. Controlled Variables
To ensure fair comparison, the following conditions should remain fixed:
Controlled VariableDecision
Dataset splitSame train/validation/test split for all experiments
Class mappingSame six-class taxonomy
Crop generationSame ground-truth crops for classification experiments
Image preprocessingSame resizing and normalization rules
Classifier training dataSame training objects
Test setSame test objects/images
Random seedFixed, for example 42
YOLO confidence thresholdFixed, for example 0.25
NMS IoU thresholdFixed, for example 0.5
## 14. Recommended Implementation Order
To make the project manageable, implement the experiments in this order:
Phase 1: Dataset and YOLO Baseline
Prepare dataset. 1.
Merge classes. 2.
Train YOLOv8-nano. 3.
Evaluate YOLO detection performance. 4.
## Output:
YOLO baseline results.
## Phase 2: Object Crop Dataset
Extract object crops from ground-truth bounding boxes. 1.
Save crop images and labels. 2.
Prepare train/validation/test crop-level datasets. 3.
## Output:
Clean crop classification dataset.
## Phase 3: Baseline Feature Models
Extract classical features. 1.
## Train Random Forest. 2.
Extract EfficientNet features. 3.
Train SVM. 4.
Compare classical-only and deep-only baselines. 5.
## Output:
Baseline representation results.
Phase 4: Early Fusion and Compression
Concatenate classical + EfficientNet features. 1.
Train raw fusion model. 2.
Apply PCA. 3.
Train PCA-fusion classifier. 4.
Train autoencoder. 5.
Extract bottleneck features. 6.
Train autoencoder-fusion classifier. 7.
Perform feature selection. 8.
Train selected-feature classifier. 9.
## Output:
Accuracy vs feature-dimension comparison.
## Phase 5: Late Fusion
Generate probability outputs from classical and deep classifiers. 1.
Apply average voting. 2.
Optimize voting weights on validation set. 3.
Evaluate on test set. 4.
## Output:
Decision-level fusion comparison.
## Phase 6: Attention Fusion
Build attention/gating network. 1.
Train on classical + deep features. 2.
Evaluate classification performance. 3.
Analyze attention weights by class. 4.
## Output:
Interpretable learned fusion model.
Phase 7: YOLO Hybrid
Extract YOLO FPN features. 1.
Combine YOLO FPN features with classical crop features. 2.
Train hybrid classifier/re-scoring module. 3.
Evaluate detection/classification performance. 4.
## Output:
Final hybrid detection result.
This is the most technically challenging experiment, so it should be done after the main feature-
fusion experiments are working.
## 15. Expected Results
The expected findings should be written carefully, not as guaranteed results.
## Expected Finding 1: Fusion Improves Classification
Classical features may perform well on materials with distinctive color and texture, such as
cardboard and paper.
Deep features may perform better on visually complex materials such as glass, metal, and
plastic.
Therefore, fusion is expected to outperform either feature type alone.
Expected Finding 2: PCA May Preserve Most Performance
PCA is expected to reduce the 2,300-dimensional fused vector significantly while preserving
most classification performance.
This would show that the full fused vector contains redundancy.
Expected Finding 3: Autoencoder May Help at Strong
## Compression
At very low dimensions, autoencoder compression may preserve nonlinear feature relationships
better than PCA.
This is useful if the final system is intended for deployment on limited hardware.
## Expected Finding 4: Attention Fusion May Be More
## Interpretable
The attention model may learn different branch weights for different waste classes.
For example:
ClassPossible Dominant Features
CardboardClassical color/texture
PaperClassical texture/color
GlassDeep visual patterns
MetalDeep + reflectiveness cues
PlasticMixed classical/deep
TrashMixed features
Expected Finding 5: YOLO Hybrid May Improve Class
## Confidence
The YOLO hybrid model may improve classification confidence by adding explicit material cues
to YOLO’s detection features.
However, it may not greatly improve localization because bounding box prediction still comes
mainly from YOLO.
- Main Risks and Solutions
RiskExplanationSolution
Too many experimentsEleven experiments can be time-
consuming
Implement in phases and
keep YOLO hybrid last
Confusion between
detection and classification
Classical/SVM models do not
detect objects by themselves
Use two evaluation tracks
YOLO FPN extraction may be
difficult
Accessing internal YOLO features
requires extra coding
Treat YOLO hybrid as
advanced/final experiment
Dataset imbalanceSome waste classes may have
fewer examples
Use macro-F1 and class
weights
Feature vector too large2,300 dimensions may be
computationally heavy
Use PCA, autoencoder, and
feature selection
OverfittingSome models may memorize
crop-level features
Use validation set, test set,
and fixed split
## 17. Final Deliverables
By the end of the project, the final submission should include:
DeliverableDescription
Dataset preprocessing codeClass merging, splitting, crop extraction
YOLO training resultsDetection baseline
Feature extraction scriptsClassical, EfficientNet, YOLO FPN
Trained classifiersRF, SVM, autoencoder, attention model
Experiment results tableAll 11 experiments compared
Confusion matricesFor classification experiments
Accuracy vs dimension chartShows compression trade-off
Speed comparisonFeature extraction and classification time
Final discussionBest model and explanation
Final report/presentationAcademic project documentation
## 18. Suggested Final Results Table Format
Your final report should include a table like this:
Exp.MethodFeature
## Dim.
## Accurac
y
## Macro-
## F1
mAP@0.5FPSModel
## Size
E1YOLOv8-nanoNative—————
E2Classical + RF252——N/A——
E3EfficientNet +
## SVM
## 1280——N/A——
E4Raw early fusion2300——N/A——
E5PCA fusion~400–600——N/A——
E6Autoencoder
fusion
## 256——N/A——
E7Feature selection200——N/A——
E8Average votingSeparate——N/A——
E9Weighted votingSeparate——N/A——
E10Attention fusion256——N/A——
E11YOLO hybrid768 + 252—————
Use N/Awhere mAP does not apply.
- Suggested Report Structure for Final Submission
When writing the final academic report, use this structure:
## Abstract 1.
## Introduction 2.
## Problem Statement 3.
## Research Question 4.
## Dataset Description 5.
## Preprocessing 6.
## Feature Extraction Methods 7.
## Fusion Strategies 8.
## Dimensionality Reduction Methods 9.
## Experimental Design 10.
## Evaluation Metrics 11.
## Results 12.
## Discussion 13.
## Limitations 14.
## Conclusion 15.
## References 16.
## 20. Final Feasibility Verdict
This project is viable and strong, as long as it is implemented with the correct separation
between detection and classification.
The most important correction is:
YOLO performs localization and detection. Classical, deep, and fused features mainly
evaluate object crop classification and representation quality.
The project should not claim that Random Forest, SVM, PCA, or attention fusion are full object
detectors unless they are integrated into a full detection pipeline with bounding boxes and
confidence scores.
The strongest final version is:
Train YOLOv8-nano as the detection baseline, then conduct a systematic representation
study on waste object crops using classical features, EfficientNet features, fusion methods,
and compression techniques. Finally, test whether YOLO’s detection features can be
improved through a hybrid fusion model.
## Project Description
Thursday, 14 May 20261:59 PM