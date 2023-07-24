# Visual-Steering-For-Deep-Neural-Networks

## Abstract
This repository contains the source code to visually steer the CNN models into alligning with the human knowledge fed into them.  This method is provided for integrating human knowledge with the convolutional neural network to improve its performance, reduce the biases that arise, and leverage human experience.

## Methodology
![methodology diagram](./figures/methodology_diagram.png)
The proposed approach is divided into 3 stages: explaining the model, manually editing the attention maps, and embedding human knowledge using a specific loss. First, Grad-CAM was used to generate attention maps describing the importance values of each image region according to the model to make a specific decision. These attention maps were then annotated by domain experts to represent what they believed were the image regions important to make that decision. Finally, the annotated attention maps were used to steer the model into aligning its behaviour with the human behaviour hence, embedding the human knowledge into the learning process of the model.

## Enviroment

## Execution

## Datasets
We applied this approach on 3 datasets:
- Imagenette2
- Breast Cancer Ultrasound Images (BUSI)
- International Skin Imaging Collaboration (ISIC)
![results](./figures/results.png)

## Models
We applied this approach on several models from the ResNet family:
- ResNet50
- ResNet101
- ResNet152

Comparison of Accuracy, AUC, F2 score, and Qualitative assessments of attention maps for Imagenette2, BUSI, and ISIC datasets. For Qualitative assessment, the lower is better, for the rest, the higher is better.

| Approach                        | Model      | Accuracy   | AUC       | Qualitative Measure | F2 Score   | Qualitative Measure | F2 Score   | Qualitative Measure |
|---------------------------------|------------|------------|-----------|---------------------|------------|---------------------|------------|---------------------|
| Without                         | ResNet50   | **99.49%** | **0.9999**| 3.4084              | 0.8298     | 1.6049              | 0.7061     | 2.9413              |
|                                 | ResNet101  | 97.73%     | 0.9995    | 3.5392              | 0.7736     | 1.8655              | 0.5715     | 3.5000              |
|                                 | ResNet152  | 98.22%     | **0.9997**| 3.2401              | 0.7860     | 1.3899              | **0.7156** | 2.9188              |
| ABN                             | ResNet50   | 97.91%     | 0.9994    | 2.3048              | 0.8238     | 2.0944              | 0.7207     | 2.2736              |
|                                 | ResNet101  | 96.46%     | 0.9982    | **2.7847**          | 0.8066     | 1.9425              | 0.6137     | **2.5259**          |
|                                 | ResNet152  | **98.32%** | **0.9997**| **2.4483**          | 0.7720     | 2.4787              | 0.6961     | **2.2733**          |
| Proposed Approach               | ResNet50   | 97.58%     | 0.9995    | **1.8623**          | **0.8599** | **1.3292**          | **0.7405** | **1.6235**          |
|                                 | ResNet101  | **98.11%** | **0.9996**| 3.1043              | **0.8429** | **1.4367**          | **0.7014** | 2.8837              |
|                                 | ResNet152  | 97.89%     | 0.9991    | 2.4673              | **0.8225** | **1.3881**          | 0.7126     | 2.8722              |

