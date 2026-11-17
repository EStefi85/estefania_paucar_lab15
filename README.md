# Brain Tumor MRI Classifier: TensorFlow Lite on Android

Lab 15 of the Artificial Intelligence course. Author: Estefania Paucar.

An Android app that classifies brain MRI images as **tumor (`yes`)** or **no tumor (`no`)** using a custom image classification model trained in Google Colab and exported to TensorFlow Lite (`.tflite`).

> **Educational project.** This model was trained on a small public dataset and must not be used for any medical or clinical decision.

## How it works

1. **Train** an image classifier in Google Colab (transfer learning with MobileNetV2).
2. **Export** the model to `model.tflite` and add metadata (labels and input normalization).
3. **Integrate** the model into the Android app (module `start`), which reads frames from the camera and shows the predicted class with its confidence.

## Model

| Item | Value |
|---|---|
| Dataset | [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) (Kaggle), 253 images, classes `no` and `yes` |
| Data split | 4 images held out as test images, remaining 249 split 80/20 (200 training, 49 validation) |
| Base model | MobileNetV2 pretrained on ImageNet, frozen |
| Head | Global average pooling, dropout 0.2, dense layer with 2 outputs (softmax) |
| Input | 224 x 224 x 3, raw pixel values 0-255 (the MobileNetV2 preprocessing is inside the model) |
| Augmentation | Random horizontal flip and random rotation |
| Training | Adam optimizer, sparse categorical cross-entropy, batch size 32, 10 epochs |
| Labels | `no`, `yes` (in this order) |

## Results

Validation accuracy: **0.8367** (the workshop requirement was above 0.7). Validation loss: 0.3313.

Confusion matrix on the 49 validation images:

| | Predicted `no` | Predicted `yes` |
|---|---|---|
| **True `no`** | 12 | 3 |
| **True `yes`** | 5 | 29 |

For the `yes` (tumor) class: precision 90.6% and recall 85.3%.

### Test images

Four images were held out from training and tested both in Colab and in the Android app. The app was tested by pointing the phone camera at the image on a computer screen, so its confidence values differ slightly from Colab.

| Image | True class | Colab | Android app |
|---|---|---|---|
| `yes_Y1.jpg` | yes | yes (98.74%) | yes (99.1%) |
| `yes_Y10.jpg` | yes | yes (89.28%) | yes (90.2%) |
| `no_1 no.jpeg` | no | no (82.26%) | no (74.3%) |
| `no_10 no.jpg` | no | no (69.18%) | no (84.3%) |

All 4 test images were classified correctly in both environments.

## Repository structure

```
start/                       Android app with the custom model integrated
  src/main/assets/
    model.tflite             Trained model with metadata
    labels.txt               Class labels (no, yes)
finish/                      Reference module from the original codelab
yes_Y1.jpg, yes_Y10.jpg,     The 4 held-out test images
no_1 no.jpeg, no_10 no.jpg
```

## Run the app

1. Open the project root folder in Android Studio and let Gradle sync.
2. Select the `start` module and run it on a physical Android device (recommended, because the app uses the camera) or on an emulator.
3. Point the camera at an MRI image and read the predicted class and confidence at the bottom of the screen.

## Limitations

- The dataset is small (253 images), so the validation set has only 49 images and each image changes the accuracy by about 2 percentage points.
- 5 of the 34 validation images with a tumor were classified as `no` (false negatives), which is the most serious kind of error in a medical setting.
- The validation accuracy is not a fully independent test. The 4 held-out images are independent but too few to estimate performance.
- Photographing a screen with a phone adds noise (lighting, angle, resolution) that the model was not trained on.

## Credits and license

The app is based on the TensorFlow Lite image classification codelab by Google: [Recognize Flowers with TensorFlow on Android](https://goo.gle/3dbCSbt). The original project structure, including the `finish` module, comes from that codelab.

```
Copyright (C) 2020 The Android Open Source Project

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
