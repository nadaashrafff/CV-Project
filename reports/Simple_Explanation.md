

Extremely Simple Explanation of the Project
Your project is about teaching a computer to recognize different types of wastelike:
## Cardboard •
## Glass •
## Metal •
## Paper •
## Plastic •
## Trash •
The main idea is:
We want to test different ways of describing waste objects to the computer, then see
which way gives the best accuracy with the smallest cost.
Your project compares normal visual features, deep learning features, and YOLO features, then
tries to combine and reduce them. This matches your project description about feature fusion
and dimensionality reduction.
- What is the problem?
A computer does not “see” an image like humans.
When we see this:
a plastic bottle
We immediately understand it is plastic.
But the computer needs numbers.
So the question is:
What numbers should we give the computer so it can recognize waste correctly?
These numbers are called features.
- What are features?
Features are simple descriptions of an object.
For example, if the object is a cardboard box, useful features could be:
FeatureSimple Meaning
ColorIs it brown, white, green, transparent?
ShapeIs it round, rectangular, flat?
TextureIs it smooth, rough, wrinkled?
EdgesDoes it have strong borders?
ShineIs it reflective like metal or glass?
So instead of giving the computer only the image, we give it useful information about the object.
- What are the three feature types in your project?
Your project uses three main types of features.
## A. Classical Features
These are features made manually using computer vision techniques.
## Examples:
## Color •
## Shape •
## Texture •
## Edges •
## Brightness •
## Reflection •
Very simple idea:
We describe the object using clear, understandable visual properties.
## Example:
A cardboard object may be recognized because it is brown, rough, and rectangular.
## B. Deep Features
These come from a pretrained neural network called EfficientNetB0.
Simple idea:
Instead of manually describing the object, we let a deep learning model extract useful
hidden patterns.
The model may notice things we cannot easily describe manually.
For example:
the shape of a bottle •
the transparency of glass •
the surface pattern of plastic •
the global appearance of paper or metal •
C. YOLO Features
YOLO is an object detection model.
It can answer:
Where is the object in the image, and what class is it?
YOLO gives:
bounding box •
class name •
confidence score •
## Example:
There is a plastic bottle here, and I am 86% confident.
In your project, YOLO is used as the main detection model.
- What does fusion mean?
Fusion means:
combining different types of features together.
Simple example:
Imagine three students are trying to identify an object.
StudentWhat they are good at
Student 1Color and texture
Student 2Deep visual patterns
Student 3Object detection
Feature fusion means we combine their opinions to get a better answer.
So instead of using only classical features or only deep features, we combine them.
- What does dimensionality reduction mean?
Features are numbers.
Sometimes we have too many numbers.
For example:
Feature TypeNumber of Features
Classical features252
Deep features2,048
Combined features2,300
That is a lot.
Dimensionality reduction means:
reducing the number of features while keeping the useful information.
Simple example:
Instead of using 2,300 numbers, maybe we can use only 256 or 500 numbers and still get almost
the same accuracy.
This makes the system:
faster •
smaller •
easier to deploy •
less computationally expensive •
- What is the main goal?
The main goal is:
Find the best way to combine and reduce features so the model can classify waste
accurately and efficiently.
So your project is not only asking:
Which model is better?
It is asking:
Which feature representation is better, and how can we make it smaller without losing
accuracy?
That is the strong research idea.
- The experiments in very simple words
You will run 11 experiments.
Each experiment tests a different way of recognizing waste.
## Group 1: Basic Experiments
These are the starting point.
E1 —YOLO Baseline
You train YOLO normally.
It looks at the full image and detects waste objects.
It gives:
object location •
object class •
confidence score •
Simple meaning:
How good is YOLO by itself?
E2 —Classical Features Only
You crop the waste object from the image.
Then you extract simple features like:
color •
shape •
texture •
edges •
Then you use Random Forest to classify it.
Simple meaning:
Can simple hand-made features recognize waste?
E3 —Deep Features Only
You crop the waste object.
Then EfficientNet extracts deep features.
Then SVM classifies the object.
Simple meaning:
Can pretrained deep features recognize waste better than classical features?
## Group 2: Early Fusion Experiments
Early fusion means:
combine the features first, then classify.
E4 —Raw Early Fusion
You combine:
classical features •
early deep features •
middle deep features •
late deep features •
Then you train a classifier.
Simple meaning:
What happens if we put all features together?
E5 —PCA Fusion
You take the big combined feature vector and reduce it using PCA.
Simple meaning:
Can we make the big feature vector smaller without losing much accuracy?
## Example:
From 2,300 features down to maybe 500 features.
E6 —Autoencoder Fusion
You use a small neural network to compress the features.
Simple meaning:
Can a neural network learn a smarter way to reduce the features?
## Example:
From 2,300 features down to 256 features.
E7 —Feature Selection Fusion
You choose only the most important features.
Simple meaning:
Do we really need all 2,300 features, or are the best 200 enough?
## Group 3: Late Fusion Experiments
Late fusion means:
train models separately first, then combine their final answers.
E8 —Average Voting
You train:
one classical feature model •
one deep feature model •
Then you average their predictions.
Simple meaning:
If both models vote together equally, does the result improve?
## Example:
Classical model says:
60% paper
Deep model says:
80% paper
Final result:
70% paper
E9 —Weighted Voting
Same as E8, but the models do not have equal power.
Maybe the deep model is stronger, so we give it more weight.
## Example:
40% classical model + 60% deep model
Simple meaning:
Should we trust one model more than the other?
## Group 4: Attention Fusion
E10 —Attention Fusion
This is a smarter fusion method.
The model learns when to trust classical features and when to trust deep features.
Simple meaning:
For each object, the model decides which feature type is more important.
## Example:
For cardboard:
Trust color and texture more.
For glass:
Trust deep features more.
For metal:
Trust shine and deep patterns.
This experiment is more advanced and gives your project a strong research part.
Group 5: YOLO Hybrid Experiment
E11 —YOLO Hybrid
This combines YOLO features with classical features.
YOLO already detects the object.
Then classical features help refine the classification.
Simple meaning:
Can YOLO become better if we give it extra color, texture, and shape information?
## Example:
YOLO detects an object and thinks it is plastic.
The classical features may help confirm or correct the class.
- Very simple summary of all experiments
ExperimentSimple Meaning
E1Test YOLO alone
E2Test classical features alone
E3Test deep features alone
E4Combine all features directly
E5Combine features, then reduce them using PCA
E6Combine features, then reduce them using autoencoder
E7Combine features, then keep only the best features
E8Let classical and deep models vote equally
E9Let classical and deep models vote with different weights
E10Let the model learn which feature type to trust
E11Combine YOLO features with classical features
- What is detection and what is classification?
This is very important.
## Detection
Detection means:
Find the object in the image and name it.
## Output:
box around the object •
class name •
confidence score •
## Example:
Plastic bottle at this location, confidence 90%.
YOLO does detection.
## Classification
Classification means:
The object is already cropped. Just tell me what it is.
## Output:
class name only •
## Example:
This cropped object is glass.
Classical features, EfficientNet features, PCA, autoencoder, voting, and attention mainly do
classification.
- What metrics will you use?
For detection experiments
Used for:
## E1 YOLO •
E11 YOLO Hybrid •
## Metrics:
MetricSimple Meaning
mAP@0.5How good the detection is
mAP@0.5:0.95Stricter detection score
PrecisionWhen the model predicts something, how often is it correct?
RecallHow many real objects did the model find?
FPSHow fast the model runs
Model sizeHow large the model file is
For classification experiments
Used for:
E2 to E10 •
## Metrics:
MetricSimple Meaning
AccuracyHow many predictions are correct overall
Macro-F1Fair score across all classes
Per-class F1Score for each waste type
Confusion matrixShows which classes get mixed up
Feature dimensionNumber of features used
TimeHow fast the method is
## Simple Explanation
Thursday, 14 May 20262:28 PM

Extremely Simple Explanation of the Project
Your project is about teaching a computer to recognize different types of wastelike:
## Cardboard •
## Glass •
## Metal •
## Paper •
## Plastic •
## Trash •
The main idea is:
We want to test different ways of describing waste objects to the computer, then see
which way gives the best accuracy with the smallest cost.
Your project compares normal visual features, deep learning features, and YOLO features, then
tries to combine and reduce them. This matches your project description about feature fusion
and dimensionality reduction.
- What is the problem?
A computer does not “see” an image like humans.
When we see this:
a plastic bottle
We immediately understand it is plastic.
But the computer needs numbers.
So the question is:
What numbers should we give the computer so it can recognize waste correctly?
These numbers are called features.
- What are features?
Features are simple descriptions of an object.
For example, if the object is a cardboard box, useful features could be:
FeatureSimple Meaning
ColorIs it brown, white, green, transparent?
ShapeIs it round, rectangular, flat?
TextureIs it smooth, rough, wrinkled?
EdgesDoes it have strong borders?
ShineIs it reflective like metal or glass?
So instead of giving the computer only the image, we give it useful information about the object.
- What are the three feature types in your project?
Your project uses three main types of features.
## A. Classical Features
These are features made manually using computer vision techniques.
## Examples:
## Color •
## Shape •
## Texture •
## Edges •
## Brightness •
## Reflection •
Very simple idea:
We describe the object using clear, understandable visual properties.
## Example:
A cardboard object may be recognized because it is brown, rough, and rectangular.
## B. Deep Features
These come from a pretrained neural network called EfficientNetB0.
Simple idea:
Instead of manually describing the object, we let a deep learning model extract useful
hidden patterns.
The model may notice things we cannot easily describe manually.
For example:
the shape of a bottle •
the transparency of glass •
the surface pattern of plastic •
the global appearance of paper or metal •
C. YOLO Features
YOLO is an object detection model.
It can answer:
Where is the object in the image, and what class is it?
YOLO gives:
bounding box •
class name •
confidence score •
## Example:
There is a plastic bottle here, and I am 86% confident.
In your project, YOLO is used as the main detection model.
- What does fusion mean?
Fusion means:
combining different types of features together.
Simple example:
Imagine three students are trying to identify an object.
StudentWhat they are good at
Student 1Color and texture
Student 2Deep visual patterns
Student 3Object detection
Feature fusion means we combine their opinions to get a better answer.
So instead of using only classical features or only deep features, we combine them.
- What does dimensionality reduction mean?
Features are numbers.
Sometimes we have too many numbers.
For example:
Feature TypeNumber of Features
Classical features252
Deep features2,048
Combined features2,300
That is a lot.
Dimensionality reduction means:
reducing the number of features while keeping the useful information.
Simple example:
Instead of using 2,300 numbers, maybe we can use only 256 or 500 numbers and still get almost
the same accuracy.
This makes the system:
faster •
smaller •
easier to deploy •
less computationally expensive •
- What is the main goal?
The main goal is:
Find the best way to combine and reduce features so the model can classify waste
accurately and efficiently.
So your project is not only asking:
Which model is better?
It is asking:
Which feature representation is better, and how can we make it smaller without losing
accuracy?
That is the strong research idea.
- The experiments in very simple words
You will run 11 experiments.
Each experiment tests a different way of recognizing waste.
## Group 1: Basic Experiments
These are the starting point.
E1 —YOLO Baseline
You train YOLO normally.
It looks at the full image and detects waste objects.
It gives:
object location •
object class •
confidence score •
Simple meaning:
How good is YOLO by itself?
E2 —Classical Features Only
You crop the waste object from the image.
Then you extract simple features like:
color •
shape •
texture •
edges •
Then you use Random Forest to classify it.
Simple meaning:
Can simple hand-made features recognize waste?
E3 —Deep Features Only
You crop the waste object.
Then EfficientNet extracts deep features.
Then SVM classifies the object.
Simple meaning:
Can pretrained deep features recognize waste better than classical features?
## Group 2: Early Fusion Experiments
Early fusion means:
combine the features first, then classify.
E4 —Raw Early Fusion
You combine:
classical features •
early deep features •
middle deep features •
late deep features •
Then you train a classifier.
Simple meaning:
What happens if we put all features together?
E5 —PCA Fusion
You take the big combined feature vector and reduce it using PCA.
Simple meaning:
Can we make the big feature vector smaller without losing much accuracy?
## Example:
From 2,300 features down to maybe 500 features.
E6 —Autoencoder Fusion
You use a small neural network to compress the features.
Simple meaning:
Can a neural network learn a smarter way to reduce the features?
## Example:
From 2,300 features down to 256 features.
E7 —Feature Selection Fusion
You choose only the most important features.
Simple meaning:
Do we really need all 2,300 features, or are the best 200 enough?
## Group 3: Late Fusion Experiments
Late fusion means:
train models separately first, then combine their final answers.
E8 —Average Voting
You train:
one classical feature model •
one deep feature model •
Then you average their predictions.
Simple meaning:
If both models vote together equally, does the result improve?
## Example:
Classical model says:
60% paper
Deep model says:
80% paper
Final result:
70% paper
E9 —Weighted Voting
Same as E8, but the models do not have equal power.
Maybe the deep model is stronger, so we give it more weight.
## Example:
40% classical model + 60% deep model
Simple meaning:
Should we trust one model more than the other?
## Group 4: Attention Fusion
E10 —Attention Fusion
This is a smarter fusion method.
The model learns when to trust classical features and when to trust deep features.
Simple meaning:
For each object, the model decides which feature type is more important.
## Example:
For cardboard:
Trust color and texture more.
For glass:
Trust deep features more.
For metal:
Trust shine and deep patterns.
This experiment is more advanced and gives your project a strong research part.
Group 5: YOLO Hybrid Experiment
E11 —YOLO Hybrid
This combines YOLO features with classical features.
YOLO already detects the object.
Then classical features help refine the classification.
Simple meaning:
Can YOLO become better if we give it extra color, texture, and shape information?
## Example:
YOLO detects an object and thinks it is plastic.
The classical features may help confirm or correct the class.
- Very simple summary of all experiments
ExperimentSimple Meaning
E1Test YOLO alone
E2Test classical features alone
E3Test deep features alone
E4Combine all features directly
E5Combine features, then reduce them using PCA
E6Combine features, then reduce them using autoencoder
E7Combine features, then keep only the best features
E8Let classical and deep models vote equally
E9Let classical and deep models vote with different weights
E10Let the model learn which feature type to trust
E11Combine YOLO features with classical features
- What is detection and what is classification?
This is very important.
## Detection
Detection means:
Find the object in the image and name it.
## Output:
box around the object •
class name •
confidence score •
## Example:
Plastic bottle at this location, confidence 90%.
YOLO does detection.
## Classification
Classification means:
The object is already cropped. Just tell me what it is.
## Output:
class name only •
## Example:
This cropped object is glass.
Classical features, EfficientNet features, PCA, autoencoder, voting, and attention mainly do
classification.
- What metrics will you use?
For detection experiments
Used for:
## E1 YOLO •
E11 YOLO Hybrid •
## Metrics:
MetricSimple Meaning
mAP@0.5How good the detection is
mAP@0.5:0.95Stricter detection score
PrecisionWhen the model predicts something, how often is it correct?
RecallHow many real objects did the model find?
FPSHow fast the model runs
Model sizeHow large the model file is
For classification experiments
Used for:
E2 to E10 •
## Metrics:
MetricSimple Meaning
AccuracyHow many predictions are correct overall
Macro-F1Fair score across all classes
Per-class F1Score for each waste type
Confusion matrixShows which classes get mixed up
Feature dimensionNumber of features used
TimeHow fast the method is
## Simple Explanation
Thursday, 14 May 20262:28 PM

Extremely Simple Explanation of the Project
Your project is about teaching a computer to recognize different types of wastelike:
## Cardboard •
## Glass •
## Metal •
## Paper •
## Plastic •
## Trash •
The main idea is:
We want to test different ways of describing waste objects to the computer, then see
which way gives the best accuracy with the smallest cost.
Your project compares normal visual features, deep learning features, and YOLO features, then
tries to combine and reduce them. This matches your project description about feature fusion
and dimensionality reduction.
- What is the problem?
A computer does not “see” an image like humans.
When we see this:
a plastic bottle
We immediately understand it is plastic.
But the computer needs numbers.
So the question is:
What numbers should we give the computer so it can recognize waste correctly?
These numbers are called features.
- What are features?
Features are simple descriptions of an object.
For example, if the object is a cardboard box, useful features could be:
FeatureSimple Meaning
ColorIs it brown, white, green, transparent?
ShapeIs it round, rectangular, flat?
TextureIs it smooth, rough, wrinkled?
EdgesDoes it have strong borders?
ShineIs it reflective like metal or glass?
So instead of giving the computer only the image, we give it useful information about the object.
- What are the three feature types in your project?
Your project uses three main types of features.
## A. Classical Features
These are features made manually using computer vision techniques.
## Examples:
## Color •
## Shape •
## Texture •
## Edges •
## Brightness •
## Reflection •
Very simple idea:
We describe the object using clear, understandable visual properties.
## Example:
A cardboard object may be recognized because it is brown, rough, and rectangular.
## B. Deep Features
These come from a pretrained neural network called EfficientNetB0.
Simple idea:
Instead of manually describing the object, we let a deep learning model extract useful
hidden patterns.
The model may notice things we cannot easily describe manually.
For example:
the shape of a bottle •
the transparency of glass •
the surface pattern of plastic •
the global appearance of paper or metal •
C. YOLO Features
YOLO is an object detection model.
It can answer:
Where is the object in the image, and what class is it?
YOLO gives:
bounding box •
class name •
confidence score •
## Example:
There is a plastic bottle here, and I am 86% confident.
In your project, YOLO is used as the main detection model.
- What does fusion mean?
Fusion means:
combining different types of features together.
Simple example:
Imagine three students are trying to identify an object.
StudentWhat they are good at
Student 1Color and texture
Student 2Deep visual patterns
Student 3Object detection
Feature fusion means we combine their opinions to get a better answer.
So instead of using only classical features or only deep features, we combine them.
- What does dimensionality reduction mean?
Features are numbers.
Sometimes we have too many numbers.
For example:
Feature TypeNumber of Features
Classical features252
Deep features2,048
Combined features2,300
That is a lot.
Dimensionality reduction means:
reducing the number of features while keeping the useful information.
Simple example:
Instead of using 2,300 numbers, maybe we can use only 256 or 500 numbers and still get almost
the same accuracy.
This makes the system:
faster •
smaller •
easier to deploy •
less computationally expensive •
- What is the main goal?
The main goal is:
Find the best way to combine and reduce features so the model can classify waste
accurately and efficiently.
So your project is not only asking:
Which model is better?
It is asking:
Which feature representation is better, and how can we make it smaller without losing
accuracy?
That is the strong research idea.
- The experiments in very simple words
You will run 11 experiments.
Each experiment tests a different way of recognizing waste.
## Group 1: Basic Experiments
These are the starting point.
E1 —YOLO Baseline
You train YOLO normally.
It looks at the full image and detects waste objects.
It gives:
object location •
object class •
confidence score •
Simple meaning:
How good is YOLO by itself?
E2 —Classical Features Only
You crop the waste object from the image.
Then you extract simple features like:
color •
shape •
texture •
edges •
Then you use Random Forest to classify it.
Simple meaning:
Can simple hand-made features recognize waste?
E3 —Deep Features Only
You crop the waste object.
Then EfficientNet extracts deep features.
Then SVM classifies the object.
Simple meaning:
Can pretrained deep features recognize waste better than classical features?
## Group 2: Early Fusion Experiments
Early fusion means:
combine the features first, then classify.
E4 —Raw Early Fusion
You combine:
classical features •
early deep features •
middle deep features •
late deep features •
Then you train a classifier.
Simple meaning:
What happens if we put all features together?
E5 —PCA Fusion
You take the big combined feature vector and reduce it using PCA.
Simple meaning:
Can we make the big feature vector smaller without losing much accuracy?
## Example:
From 2,300 features down to maybe 500 features.
E6 —Autoencoder Fusion
You use a small neural network to compress the features.
Simple meaning:
Can a neural network learn a smarter way to reduce the features?
## Example:
From 2,300 features down to 256 features.
E7 —Feature Selection Fusion
You choose only the most important features.
Simple meaning:
Do we really need all 2,300 features, or are the best 200 enough?
## Group 3: Late Fusion Experiments
Late fusion means:
train models separately first, then combine their final answers.
E8 —Average Voting
You train:
one classical feature model •
one deep feature model •
Then you average their predictions.
Simple meaning:
If both models vote together equally, does the result improve?
## Example:
Classical model says:
60% paper
Deep model says:
80% paper
Final result:
70% paper
E9 —Weighted Voting
Same as E8, but the models do not have equal power.
Maybe the deep model is stronger, so we give it more weight.
## Example:
40% classical model + 60% deep model
Simple meaning:
Should we trust one model more than the other?
## Group 4: Attention Fusion
E10 —Attention Fusion
This is a smarter fusion method.
The model learns when to trust classical features and when to trust deep features.
Simple meaning:
For each object, the model decides which feature type is more important.
## Example:
For cardboard:
Trust color and texture more.
For glass:
Trust deep features more.
For metal:
Trust shine and deep patterns.
This experiment is more advanced and gives your project a strong research part.
Group 5: YOLO Hybrid Experiment
E11 —YOLO Hybrid
This combines YOLO features with classical features.
YOLO already detects the object.
Then classical features help refine the classification.
Simple meaning:
Can YOLO become better if we give it extra color, texture, and shape information?
## Example:
YOLO detects an object and thinks it is plastic.
The classical features may help confirm or correct the class.
- Very simple summary of all experiments
ExperimentSimple Meaning
E1Test YOLO alone
E2Test classical features alone
E3Test deep features alone
E4Combine all features directly
E5Combine features, then reduce them using PCA
E6Combine features, then reduce them using autoencoder
E7Combine features, then keep only the best features
E8Let classical and deep models vote equally
E9Let classical and deep models vote with different weights
E10Let the model learn which feature type to trust
E11Combine YOLO features with classical features
- What is detection and what is classification?
This is very important.
## Detection
Detection means:
Find the object in the image and name it.
## Output:
box around the object •
class name •
confidence score •
## Example:
Plastic bottle at this location, confidence 90%.
YOLO does detection.
## Classification
Classification means:
The object is already cropped. Just tell me what it is.
## Output:
class name only •
## Example:
This cropped object is glass.
Classical features, EfficientNet features, PCA, autoencoder, voting, and attention mainly do
classification.
- What metrics will you use?
For detection experiments
Used for:
## E1 YOLO •
E11 YOLO Hybrid •
## Metrics:
MetricSimple Meaning
mAP@0.5How good the detection is
mAP@0.5:0.95Stricter detection score
PrecisionWhen the model predicts something, how often is it correct?
RecallHow many real objects did the model find?
FPSHow fast the model runs
Model sizeHow large the model file is
For classification experiments
Used for:
E2 to E10 •
## Metrics:
MetricSimple Meaning
AccuracyHow many predictions are correct overall
Macro-F1Fair score across all classes
Per-class F1Score for each waste type
Confusion matrixShows which classes get mixed up
Feature dimensionNumber of features used
TimeHow fast the method is
## Simple Explanation
Thursday, 14 May 20262:28 PM

Extremely Simple Explanation of the Project
Your project is about teaching a computer to recognize different types of wastelike:
## Cardboard •
## Glass •
## Metal •
## Paper •
## Plastic •
## Trash •
The main idea is:
We want to test different ways of describing waste objects to the computer, then see
which way gives the best accuracy with the smallest cost.
Your project compares normal visual features, deep learning features, and YOLO features, then
tries to combine and reduce them. This matches your project description about feature fusion
and dimensionality reduction.
- What is the problem?
A computer does not “see” an image like humans.
When we see this:
a plastic bottle
We immediately understand it is plastic.
But the computer needs numbers.
So the question is:
What numbers should we give the computer so it can recognize waste correctly?
These numbers are called features.
- What are features?
Features are simple descriptions of an object.
For example, if the object is a cardboard box, useful features could be:
FeatureSimple Meaning
ColorIs it brown, white, green, transparent?
ShapeIs it round, rectangular, flat?
TextureIs it smooth, rough, wrinkled?
EdgesDoes it have strong borders?
ShineIs it reflective like metal or glass?
So instead of giving the computer only the image, we give it useful information about the object.
- What are the three feature types in your project?
Your project uses three main types of features.
## A. Classical Features
These are features made manually using computer vision techniques.
## Examples:
## Color •
## Shape •
## Texture •
## Edges •
## Brightness •
## Reflection •
Very simple idea:
We describe the object using clear, understandable visual properties.
## Example:
A cardboard object may be recognized because it is brown, rough, and rectangular.
## B. Deep Features
These come from a pretrained neural network called EfficientNetB0.
Simple idea:
Instead of manually describing the object, we let a deep learning model extract useful
hidden patterns.
The model may notice things we cannot easily describe manually.
For example:
the shape of a bottle •
the transparency of glass •
the surface pattern of plastic •
the global appearance of paper or metal •
C. YOLO Features
YOLO is an object detection model.
It can answer:
Where is the object in the image, and what class is it?
YOLO gives:
bounding box •
class name •
confidence score •
## Example:
There is a plastic bottle here, and I am 86% confident.
In your project, YOLO is used as the main detection model.
- What does fusion mean?
Fusion means:
combining different types of features together.
Simple example:
Imagine three students are trying to identify an object.
StudentWhat they are good at
Student 1Color and texture
Student 2Deep visual patterns
Student 3Object detection
Feature fusion means we combine their opinions to get a better answer.
So instead of using only classical features or only deep features, we combine them.
- What does dimensionality reduction mean?
Features are numbers.
Sometimes we have too many numbers.
For example:
Feature TypeNumber of Features
Classical features252
Deep features2,048
Combined features2,300
That is a lot.
Dimensionality reduction means:
reducing the number of features while keeping the useful information.
Simple example:
Instead of using 2,300 numbers, maybe we can use only 256 or 500 numbers and still get almost
the same accuracy.
This makes the system:
faster •
smaller •
easier to deploy •
less computationally expensive •
- What is the main goal?
The main goal is:
Find the best way to combine and reduce features so the model can classify waste
accurately and efficiently.
So your project is not only asking:
Which model is better?
It is asking:
Which feature representation is better, and how can we make it smaller without losing
accuracy?
That is the strong research idea.
- The experiments in very simple words
You will run 11 experiments.
Each experiment tests a different way of recognizing waste.
## Group 1: Basic Experiments
These are the starting point.
E1 —YOLO Baseline
You train YOLO normally.
It looks at the full image and detects waste objects.
It gives:
object location •
object class •
confidence score •
Simple meaning:
How good is YOLO by itself?
E2 —Classical Features Only
You crop the waste object from the image.
Then you extract simple features like:
color •
shape •
texture •
edges •
Then you use Random Forest to classify it.
Simple meaning:
Can simple hand-made features recognize waste?
E3 —Deep Features Only
You crop the waste object.
Then EfficientNet extracts deep features.
Then SVM classifies the object.
Simple meaning:
Can pretrained deep features recognize waste better than classical features?
## Group 2: Early Fusion Experiments
Early fusion means:
combine the features first, then classify.
E4 —Raw Early Fusion
You combine:
classical features •
early deep features •
middle deep features •
late deep features •
Then you train a classifier.
Simple meaning:
What happens if we put all features together?
E5 —PCA Fusion
You take the big combined feature vector and reduce it using PCA.
Simple meaning:
Can we make the big feature vector smaller without losing much accuracy?
## Example:
From 2,300 features down to maybe 500 features.
E6 —Autoencoder Fusion
You use a small neural network to compress the features.
Simple meaning:
Can a neural network learn a smarter way to reduce the features?
## Example:
From 2,300 features down to 256 features.
E7 —Feature Selection Fusion
You choose only the most important features.
Simple meaning:
Do we really need all 2,300 features, or are the best 200 enough?
## Group 3: Late Fusion Experiments
Late fusion means:
train models separately first, then combine their final answers.
E8 —Average Voting
You train:
one classical feature model •
one deep feature model •
Then you average their predictions.
Simple meaning:
If both models vote together equally, does the result improve?
## Example:
Classical model says:
60% paper
Deep model says:
80% paper
Final result:
70% paper
E9 —Weighted Voting
Same as E8, but the models do not have equal power.
Maybe the deep model is stronger, so we give it more weight.
## Example:
40% classical model + 60% deep model
Simple meaning:
Should we trust one model more than the other?
## Group 4: Attention Fusion
E10 —Attention Fusion
This is a smarter fusion method.
The model learns when to trust classical features and when to trust deep features.
Simple meaning:
For each object, the model decides which feature type is more important.
## Example:
For cardboard:
Trust color and texture more.
For glass:
Trust deep features more.
For metal:
Trust shine and deep patterns.
This experiment is more advanced and gives your project a strong research part.
Group 5: YOLO Hybrid Experiment
E11 —YOLO Hybrid
This combines YOLO features with classical features.
YOLO already detects the object.
Then classical features help refine the classification.
Simple meaning:
Can YOLO become better if we give it extra color, texture, and shape information?
## Example:
YOLO detects an object and thinks it is plastic.
The classical features may help confirm or correct the class.
- Very simple summary of all experiments
ExperimentSimple Meaning
E1Test YOLO alone
E2Test classical features alone
E3Test deep features alone
E4Combine all features directly
E5Combine features, then reduce them using PCA
E6Combine features, then reduce them using autoencoder
E7Combine features, then keep only the best features
E8Let classical and deep models vote equally
E9Let classical and deep models vote with different weights
E10Let the model learn which feature type to trust
E11Combine YOLO features with classical features
- What is detection and what is classification?
This is very important.
## Detection
Detection means:
Find the object in the image and name it.
## Output:
box around the object •
class name •
confidence score •
## Example:
Plastic bottle at this location, confidence 90%.
YOLO does detection.
## Classification
Classification means:
The object is already cropped. Just tell me what it is.
## Output:
class name only •
## Example:
This cropped object is glass.
Classical features, EfficientNet features, PCA, autoencoder, voting, and attention mainly do
classification.
- What metrics will you use?
For detection experiments
Used for:
## E1 YOLO •
E11 YOLO Hybrid •
## Metrics:
MetricSimple Meaning
mAP@0.5How good the detection is
mAP@0.5:0.95Stricter detection score
PrecisionWhen the model predicts something, how often is it correct?
RecallHow many real objects did the model find?
FPSHow fast the model runs
Model sizeHow large the model file is
For classification experiments
Used for:
E2 to E10 •
## Metrics:
MetricSimple Meaning
AccuracyHow many predictions are correct overall
Macro-F1Fair score across all classes
Per-class F1Score for each waste type
Confusion matrixShows which classes get mixed up
Feature dimensionNumber of features used
TimeHow fast the method is
## Simple Explanation
Thursday, 14 May 20262:28 PM

Extremely Simple Explanation of the Project
Your project is about teaching a computer to recognize different types of wastelike:
## Cardboard •
## Glass •
## Metal •
## Paper •
## Plastic •
## Trash •
The main idea is:
We want to test different ways of describing waste objects to the computer, then see
which way gives the best accuracy with the smallest cost.
Your project compares normal visual features, deep learning features, and YOLO features, then
tries to combine and reduce them. This matches your project description about feature fusion
and dimensionality reduction.
- What is the problem?
A computer does not “see” an image like humans.
When we see this:
a plastic bottle
We immediately understand it is plastic.
But the computer needs numbers.
So the question is:
What numbers should we give the computer so it can recognize waste correctly?
These numbers are called features.
- What are features?
Features are simple descriptions of an object.
For example, if the object is a cardboard box, useful features could be:
FeatureSimple Meaning
ColorIs it brown, white, green, transparent?
ShapeIs it round, rectangular, flat?
TextureIs it smooth, rough, wrinkled?
EdgesDoes it have strong borders?
ShineIs it reflective like metal or glass?
So instead of giving the computer only the image, we give it useful information about the object.
- What are the three feature types in your project?
Your project uses three main types of features.
## A. Classical Features
These are features made manually using computer vision techniques.
## Examples:
## Color •
## Shape •
## Texture •
## Edges •
## Brightness •
## Reflection •
Very simple idea:
We describe the object using clear, understandable visual properties.
## Example:
A cardboard object may be recognized because it is brown, rough, and rectangular.
## B. Deep Features
These come from a pretrained neural network called EfficientNetB0.
Simple idea:
Instead of manually describing the object, we let a deep learning model extract useful
hidden patterns.
The model may notice things we cannot easily describe manually.
For example:
the shape of a bottle •
the transparency of glass •
the surface pattern of plastic •
the global appearance of paper or metal •
C. YOLO Features
YOLO is an object detection model.
It can answer:
Where is the object in the image, and what class is it?
YOLO gives:
bounding box •
class name •
confidence score •
## Example:
There is a plastic bottle here, and I am 86% confident.
In your project, YOLO is used as the main detection model.
- What does fusion mean?
Fusion means:
combining different types of features together.
Simple example:
Imagine three students are trying to identify an object.
StudentWhat they are good at
Student 1Color and texture
Student 2Deep visual patterns
Student 3Object detection
Feature fusion means we combine their opinions to get a better answer.
So instead of using only classical features or only deep features, we combine them.
- What does dimensionality reduction mean?
Features are numbers.
Sometimes we have too many numbers.
For example:
Feature TypeNumber of Features
Classical features252
Deep features2,048
Combined features2,300
That is a lot.
Dimensionality reduction means:
reducing the number of features while keeping the useful information.
Simple example:
Instead of using 2,300 numbers, maybe we can use only 256 or 500 numbers and still get almost
the same accuracy.
This makes the system:
faster •
smaller •
easier to deploy •
less computationally expensive •
- What is the main goal?
The main goal is:
Find the best way to combine and reduce features so the model can classify waste
accurately and efficiently.
So your project is not only asking:
Which model is better?
It is asking:
Which feature representation is better, and how can we make it smaller without losing
accuracy?
That is the strong research idea.
- The experiments in very simple words
You will run 11 experiments.
Each experiment tests a different way of recognizing waste.
## Group 1: Basic Experiments
These are the starting point.
E1 —YOLO Baseline
You train YOLO normally.
It looks at the full image and detects waste objects.
It gives:
object location •
object class •
confidence score •
Simple meaning:
How good is YOLO by itself?
E2 —Classical Features Only
You crop the waste object from the image.
Then you extract simple features like:
color •
shape •
texture •
edges •
Then you use Random Forest to classify it.
Simple meaning:
Can simple hand-made features recognize waste?
E3 —Deep Features Only
You crop the waste object.
Then EfficientNet extracts deep features.
Then SVM classifies the object.
Simple meaning:
Can pretrained deep features recognize waste better than classical features?
## Group 2: Early Fusion Experiments
Early fusion means:
combine the features first, then classify.
E4 —Raw Early Fusion
You combine:
classical features •
early deep features •
middle deep features •
late deep features •
Then you train a classifier.
Simple meaning:
What happens if we put all features together?
E5 —PCA Fusion
You take the big combined feature vector and reduce it using PCA.
Simple meaning:
Can we make the big feature vector smaller without losing much accuracy?
## Example:
From 2,300 features down to maybe 500 features.
E6 —Autoencoder Fusion
You use a small neural network to compress the features.
Simple meaning:
Can a neural network learn a smarter way to reduce the features?
## Example:
From 2,300 features down to 256 features.
E7 —Feature Selection Fusion
You choose only the most important features.
Simple meaning:
Do we really need all 2,300 features, or are the best 200 enough?
## Group 3: Late Fusion Experiments
Late fusion means:
train models separately first, then combine their final answers.
E8 —Average Voting
You train:
one classical feature model •
one deep feature model •
Then you average their predictions.
Simple meaning:
If both models vote together equally, does the result improve?
## Example:
Classical model says:
60% paper
Deep model says:
80% paper
Final result:
70% paper
E9 —Weighted Voting
Same as E8, but the models do not have equal power.
Maybe the deep model is stronger, so we give it more weight.
## Example:
40% classical model + 60% deep model
Simple meaning:
Should we trust one model more than the other?
## Group 4: Attention Fusion
E10 —Attention Fusion
This is a smarter fusion method.
The model learns when to trust classical features and when to trust deep features.
Simple meaning:
For each object, the model decides which feature type is more important.
## Example:
For cardboard:
Trust color and texture more.
For glass:
Trust deep features more.
For metal:
Trust shine and deep patterns.
This experiment is more advanced and gives your project a strong research part.
Group 5: YOLO Hybrid Experiment
E11 —YOLO Hybrid
This combines YOLO features with classical features.
YOLO already detects the object.
Then classical features help refine the classification.
Simple meaning:
Can YOLO become better if we give it extra color, texture, and shape information?
## Example:
YOLO detects an object and thinks it is plastic.
The classical features may help confirm or correct the class.
- Very simple summary of all experiments
ExperimentSimple Meaning
E1Test YOLO alone
E2Test classical features alone
E3Test deep features alone
E4Combine all features directly
E5Combine features, then reduce them using PCA
E6Combine features, then reduce them using autoencoder
E7Combine features, then keep only the best features
E8Let classical and deep models vote equally
E9Let classical and deep models vote with different weights
E10Let the model learn which feature type to trust
E11Combine YOLO features with classical features
- What is detection and what is classification?
This is very important.
## Detection
Detection means:
Find the object in the image and name it.
## Output:
box around the object •
class name •
confidence score •
## Example:
Plastic bottle at this location, confidence 90%.
YOLO does detection.
## Classification
Classification means:
The object is already cropped. Just tell me what it is.
## Output:
class name only •
## Example:
This cropped object is glass.
Classical features, EfficientNet features, PCA, autoencoder, voting, and attention mainly do
classification.
- What metrics will you use?
For detection experiments
Used for:
## E1 YOLO •
E11 YOLO Hybrid •
## Metrics:
MetricSimple Meaning
mAP@0.5How good the detection is
mAP@0.5:0.95Stricter detection score
PrecisionWhen the model predicts something, how often is it correct?
RecallHow many real objects did the model find?
FPSHow fast the model runs
Model sizeHow large the model file is
For classification experiments
Used for:
E2 to E10 •
## Metrics:
MetricSimple Meaning
AccuracyHow many predictions are correct overall
Macro-F1Fair score across all classes
Per-class F1Score for each waste type
Confusion matrixShows which classes get mixed up
Feature dimensionNumber of features used
TimeHow fast the method is
## Simple Explanation
Thursday, 14 May 20262:28 PM

Extremely Simple Explanation of the Project
Your project is about teaching a computer to recognize different types of wastelike:
## Cardboard •
## Glass •
## Metal •
## Paper •
## Plastic •
## Trash •
The main idea is:
We want to test different ways of describing waste objects to the computer, then see
which way gives the best accuracy with the smallest cost.
Your project compares normal visual features, deep learning features, and YOLO features, then
tries to combine and reduce them. This matches your project description about feature fusion
and dimensionality reduction.
- What is the problem?
A computer does not “see” an image like humans.
When we see this:
a plastic bottle
We immediately understand it is plastic.
But the computer needs numbers.
So the question is:
What numbers should we give the computer so it can recognize waste correctly?
These numbers are called features.
- What are features?
Features are simple descriptions of an object.
For example, if the object is a cardboard box, useful features could be:
FeatureSimple Meaning
ColorIs it brown, white, green, transparent?
ShapeIs it round, rectangular, flat?
TextureIs it smooth, rough, wrinkled?
EdgesDoes it have strong borders?
ShineIs it reflective like metal or glass?
So instead of giving the computer only the image, we give it useful information about the object.
- What are the three feature types in your project?
Your project uses three main types of features.
## A. Classical Features
These are features made manually using computer vision techniques.
## Examples:
## Color •
## Shape •
## Texture •
## Edges •
## Brightness •
## Reflection •
Very simple idea:
We describe the object using clear, understandable visual properties.
## Example:
A cardboard object may be recognized because it is brown, rough, and rectangular.
## B. Deep Features
These come from a pretrained neural network called EfficientNetB0.
Simple idea:
Instead of manually describing the object, we let a deep learning model extract useful
hidden patterns.
The model may notice things we cannot easily describe manually.
For example:
the shape of a bottle •
the transparency of glass •
the surface pattern of plastic •
the global appearance of paper or metal •
C. YOLO Features
YOLO is an object detection model.
It can answer:
Where is the object in the image, and what class is it?
YOLO gives:
bounding box •
class name •
confidence score •
## Example:
There is a plastic bottle here, and I am 86% confident.
In your project, YOLO is used as the main detection model.
- What does fusion mean?
Fusion means:
combining different types of features together.
Simple example:
Imagine three students are trying to identify an object.
StudentWhat they are good at
Student 1Color and texture
Student 2Deep visual patterns
Student 3Object detection
Feature fusion means we combine their opinions to get a better answer.
So instead of using only classical features or only deep features, we combine them.
- What does dimensionality reduction mean?
Features are numbers.
Sometimes we have too many numbers.
For example:
Feature TypeNumber of Features
Classical features252
Deep features2,048
Combined features2,300
That is a lot.
Dimensionality reduction means:
reducing the number of features while keeping the useful information.
Simple example:
Instead of using 2,300 numbers, maybe we can use only 256 or 500 numbers and still get almost
the same accuracy.
This makes the system:
faster •
smaller •
easier to deploy •
less computationally expensive •
- What is the main goal?
The main goal is:
Find the best way to combine and reduce features so the model can classify waste
accurately and efficiently.
So your project is not only asking:
Which model is better?
It is asking:
Which feature representation is better, and how can we make it smaller without losing
accuracy?
That is the strong research idea.
- The experiments in very simple words
You will run 11 experiments.
Each experiment tests a different way of recognizing waste.
## Group 1: Basic Experiments
These are the starting point.
E1 —YOLO Baseline
You train YOLO normally.
It looks at the full image and detects waste objects.
It gives:
object location •
object class •
confidence score •
Simple meaning:
How good is YOLO by itself?
E2 —Classical Features Only
You crop the waste object from the image.
Then you extract simple features like:
color •
shape •
texture •
edges •
Then you use Random Forest to classify it.
Simple meaning:
Can simple hand-made features recognize waste?
E3 —Deep Features Only
You crop the waste object.
Then EfficientNet extracts deep features.
Then SVM classifies the object.
Simple meaning:
Can pretrained deep features recognize waste better than classical features?
## Group 2: Early Fusion Experiments
Early fusion means:
combine the features first, then classify.
E4 —Raw Early Fusion
You combine:
classical features •
early deep features •
middle deep features •
late deep features •
Then you train a classifier.
Simple meaning:
What happens if we put all features together?
E5 —PCA Fusion
You take the big combined feature vector and reduce it using PCA.
Simple meaning:
Can we make the big feature vector smaller without losing much accuracy?
## Example:
From 2,300 features down to maybe 500 features.
E6 —Autoencoder Fusion
You use a small neural network to compress the features.
Simple meaning:
Can a neural network learn a smarter way to reduce the features?
## Example:
From 2,300 features down to 256 features.
E7 —Feature Selection Fusion
You choose only the most important features.
Simple meaning:
Do we really need all 2,300 features, or are the best 200 enough?
## Group 3: Late Fusion Experiments
Late fusion means:
train models separately first, then combine their final answers.
E8 —Average Voting
You train:
one classical feature model •
one deep feature model •
Then you average their predictions.
Simple meaning:
If both models vote together equally, does the result improve?
## Example:
Classical model says:
60% paper
Deep model says:
80% paper
Final result:
70% paper
E9 —Weighted Voting
Same as E8, but the models do not have equal power.
Maybe the deep model is stronger, so we give it more weight.
## Example:
40% classical model + 60% deep model
Simple meaning:
Should we trust one model more than the other?
## Group 4: Attention Fusion
E10 —Attention Fusion
This is a smarter fusion method.
The model learns when to trust classical features and when to trust deep features.
Simple meaning:
For each object, the model decides which feature type is more important.
## Example:
For cardboard:
Trust color and texture more.
For glass:
Trust deep features more.
For metal:
Trust shine and deep patterns.
This experiment is more advanced and gives your project a strong research part.
Group 5: YOLO Hybrid Experiment
E11 —YOLO Hybrid
This combines YOLO features with classical features.
YOLO already detects the object.
Then classical features help refine the classification.
Simple meaning:
Can YOLO become better if we give it extra color, texture, and shape information?
## Example:
YOLO detects an object and thinks it is plastic.
The classical features may help confirm or correct the class.
- Very simple summary of all experiments
ExperimentSimple Meaning
E1Test YOLO alone
E2Test classical features alone
E3Test deep features alone
E4Combine all features directly
E5Combine features, then reduce them using PCA
E6Combine features, then reduce them using autoencoder
E7Combine features, then keep only the best features
E8Let classical and deep models vote equally
E9Let classical and deep models vote with different weights
E10Let the model learn which feature type to trust
E11Combine YOLO features with classical features
- What is detection and what is classification?
This is very important.
## Detection
Detection means:
Find the object in the image and name it.
## Output:
box around the object •
class name •
confidence score •
## Example:
Plastic bottle at this location, confidence 90%.
YOLO does detection.
## Classification
Classification means:
The object is already cropped. Just tell me what it is.
## Output:
class name only •
## Example:
This cropped object is glass.
Classical features, EfficientNet features, PCA, autoencoder, voting, and attention mainly do
classification.
- What metrics will you use?
For detection experiments
Used for:
## E1 YOLO •
E11 YOLO Hybrid •
## Metrics:
MetricSimple Meaning
mAP@0.5How good the detection is
mAP@0.5:0.95Stricter detection score
PrecisionWhen the model predicts something, how often is it correct?
RecallHow many real objects did the model find?
FPSHow fast the model runs
Model sizeHow large the model file is
For classification experiments
Used for:
E2 to E10 •
## Metrics:
MetricSimple Meaning
AccuracyHow many predictions are correct overall
Macro-F1Fair score across all classes
Per-class F1Score for each waste type
Confusion matrixShows which classes get mixed up
Feature dimensionNumber of features used
TimeHow fast the method is
## Simple Explanation
Thursday, 14 May 20262:28 PM