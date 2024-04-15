# Bridge-Classifier
This project aims to build an image classifier using deep learning techniques to distinguish between suspension bridges and arch bridges based on input images. The project utilizes TensorFlow and Keras libraries for model development and implementation. The data set containing 800+ images of bridges were scrapped from google images with the aid of the "Download All Images" extension.

## Data Preprocessing

Initially, the dataset undergoes preprocessing steps, including the removal of potentially corrupted images and data augmentation techniques to enhance the robustness of the model. Images with incompatible file extensions ('jpeg', 'jpg', 'bmp', 'png') are excluded from the dataset to ensure data integrity. The resulting dataset has a total of 800 images, 400 assign to each of the categories ("arch bridge" and "suspension bridge").

> Suspension Bridge Class: 1 <br>
> Arch Bridge Class: 0<br>

<p align="center">
  <img width="1000" height="350" src="https://github.com/gomeslelino/Bridge-Classifier/blob/main/Pictures/BatchVisualization.png">
</p>

## Data Splitting and Model Architecture

After preprocessing, the dataset is split into three subsets: training, validation, and testing sets. The training set comprises 70% of the data, while the validation and testing sets consist of 20% and 10%, respectively. This partitioning ensures adequate data for model training, validation, and evaluation.

The model architecture is designed using a Convolutional Neural Network (CNN) approach. The architecture consists of convolutional layers followed by max-pooling layers to extract relevant features from input images. Dropout layers are incorporated to prevent overfitting, and densely connected layers are added for classification.

## Model Compilation and Training

Before training, the model is compiled with the Adam optimizer and the Binary Crossentropy loss function. Additionally, accuracy is chosen as the evaluation metric to assess the model's performance during training.

The compiled model is trained using the training dataset for a specified number of epochs. During training, the model learns to classify input images into either suspension or arch bridges based on the features extracted by the CNN layers. The training process is monitored for both training and validation performance.

The results were plotted in two graphs:

> Training and Validation Loss: This graph visualizes how the loss (error) changes over epochs for both the training and validation sets. It helps to assess whether the model is overfitting or underfitting by comparing the loss values between the training and validation sets. 

<p align="center">
  <img width="567" height="453" src="https://github.com/gomeslelino/Bridge-Classifier/blob/main/Pictures/ModelLoss.png">
</p>

The Validation Loss was decreasing toward zero in a different rate than Training Loss. Results achieved after some parameter fine tuning.

> Training and Validation Accuracy: This graph shows how the accuracy of the model changes over epochs for both the training and validation sets. It provides insights into the model's ability to generalize to unseen data.

<p align="center">
  <img width="567" height="453" src="https://github.com/gomeslelino/Bridge-Classifier/blob/main/Pictures/ModelAccuracy.png">
</p>

Increasing accuracy on both sets indicates that the model is learning useful patterns from the training data and can generalize well to new data. Some iterations were run by calibrating the parameters and the best result achieved was presented above.

## Model Evaluation

Following training, the model's performance is evaluated using the testing dataset. Performance metrics such as accuracy, precision, and recall are computed to assess the model's ability to correctly classify bridge types. The results were very satisfactory (practically 1 in all factors):

> Accuracy: 1.0<br>
> Precision: 0.97<br>
> Recall: 0.98 <br>

## Model Deployment

Upon successful evaluation, the trained model is saved for future use. It can be deployed to classify bridge types in real-world scenarios, providing valuable insights for civil engineering applications.
The model was tested with two images (one arch and one suspension) not yet presented to the model.

<p align="center">
  <img width="900" height="400" src="https://github.com/gomeslelino/Bridge-Classifier/blob/main/Pictures/SuspensionTest.png">
</p>

For the suspension bridge picture, the model returned a result of 0.2798247, which is lower 0,5000, which means the bridge was correctly classified as suspension.

<p align="center">
  <img width="900" height="400" src="https://github.com/gomeslelino/Bridge-Classifier/blob/main/Pictures/ArchTest.png">
</p>

For the suspension bridge picture, the model returned a result of 0.9999979, which is higher than 0,5000 and pratically 1,000, which means the bridge was correctly classified as arch.

## Conclusion
In conclusion, this project demonstrates the application of deep learning techniques for image classification tasks in civil engineering. By accurately distinguishing between suspension and arch bridges, the model offers potential benefits in bridge type recognition and infrastructure management.
