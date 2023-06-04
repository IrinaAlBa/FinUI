# FinUI

### DE4 Master's Project: Automation of the User Interface Design Evaluation process in Financial Applications

The aesthetics of a user interface (UI) significantly influence impressions and further engagement, inspiring interest to quantify and predict aesthetic appeal using modern machine learning tools.
While there have been fascinating developments in the area, there is still a need for improvement in consideration of industry-based expectations, key UI component evaluation 
and explainability of the obtained solutions.
The present study focuses on financial UI, offering a novel dataset of 100 UI images with the corresponding expert ratings collected via survey evaluating different aspects of design.
Three types of models are considered: linear regression and gradient boosting machines, trained using 51 hand-crafted features derived from contextual and aesthetic qualities of a UI filtered using mutual information, and convolutional neural network applied directly to raw UI images boosted by transfer learning.
A linear model was found to be preferable for the current dataset.
A software solution is provided automating the process, from UI element recognition to feature calculation to model prediction yielding a range of explainable aesthetic score components.

Repository follows research topics discussed in the report, requirements are provided where necessary.
Dash app is used to visualize the results.
