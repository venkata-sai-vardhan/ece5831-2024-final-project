# Detecting Generative AI Images vs. Original Images Using Neural Networks

The Generative AI vs Original images project is designed to address misinformation issue caused by generative AI. Gen AI model are designed to generate new human-like content based on the patterns learned from the dataset. It opened up many opportunities for creativity and innovation. It also comes with a set of significant challenges like misinformation, job displacement, copyright issues, bias and more. This project focuses on training a neural network to differentiate between generative AI images and original images. This addresses the growing challenge faced in domains such as digital forensics, misinformation detection, and intellectual property protection.

## Installation

The output of this project is a keras model. In order to use the model you need python installed in you system with tensorflow, keras, os, shutil, PIL, numpy modules.

If you already have python installed, use below commands to install required modules if not installed.

```bash
pip install tensorflow
pip install os
pip install shutil
pip install PIL
pip install numpy
```

## Usage

In order to use the model(genai_vs_original_model.keras), first we have load it. Use below code to load keras model.

```python
Keras_model = tensorflow.keras.models.load_model('genai_vs_original_model.keras')
```
Before sending image to our model, we have preprocess it.

```python
img = load_img(<path to image>, target_size=(128, 128))
img_array = img_to_array(img) / 255.0  # Normalize the image
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
```
Now we can send our preprocessed image to model.
```python
prediction = keras_model.predict(img_array)
```
This will return the prediction as a floating point number between 0 and 1. If the prediction is greater that 0.5 then it's a original image else it's a GenAI image.

## Resources

Please follow the below links for more details and for the resources used in this project.

1. [Presentation](https://www.youtube.com/)
2. [Presentation slides](https://drive.google.com/drive/folders/1qkOowiAIAxT5FCGa9zPCnAdUYZLIaSvf?usp=drive_link)
3. [Report](https://drive.google.com/drive/folders/1EY1eXZzhzGRZ3hmWlTA-qHdtICuhy4Rm?usp=drive_link)
4. [Dataset](https://drive.google.com/drive/folders/1jbgUeYK5d7jNre56BXjcqfuunMPHeVR2?usp=drive_link)
5. [Demo](https://www.youtube.com/)

## Credits

This project is done by following students at university of Michigan-Dearborn.
- Muralidhar Pothugunta
- Venkata Sai Vardhan Velivela
- Kirubakkar Ravichandran
