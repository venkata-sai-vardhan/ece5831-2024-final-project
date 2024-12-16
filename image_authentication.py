import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.metrics import classification_report, f1_score # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
import matplotlib.pyplot as plt
import os
import numpy as np

class ImageAuthenticityClassifier:
    def __init__(self, img_height=128, img_width=128, batch_size=32, epochs=20):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.history = None

    def prepare_dataset(self, train_dir, val_dir, test_dir):
        """
        Prepare the datasets for training, validation, and testing.
        """
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
        )
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="binary",
        )
        self.val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="binary",
        )
        self.test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="binary",
            shuffle=False,
        )

    def build(self):
        """
        Build the CNN model.
        """
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['accuracy'])
        print(self.model.summary())

    def train(self):
        """
        Train the model.
        """
        self.history = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.val_generator
        )
    def load_model(self, model_name):
        self.model = load_model(model_name)
        print(f"Model is loaded from the path {model_name}")

    def save_model(self, model_path):
        """
        Save the trained model.
        """
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def plot_loss(self):
        """
        Plot the training and validation loss.
        """
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.show()

    def plot_accuracy(self):
        """
        Plot the training and validation accuracy.
        """
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.show()

    def evaluate(self):
        """
        Evaluate the model on the test dataset.
        """
        test_loss, test_accuracy = self.model.evaluate(self.test_generator)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Compute F1 Score
        y_true = self.test_generator.classes
        y_pred = (self.model.predict(self.test_generator) > 0.5).astype(int).flatten()
        f1 = f1_score(y_true, y_pred)
        print(f"F1 Score: {f1:.4f}")

        # Classification Report
        report = classification_report(y_true, y_pred, target_names=list(self.test_generator.class_indices.keys()))
        print("\nClassification Report:\n", report)

    def predict_image(self,image_path):
        """
        Preprocess an input image and predict its class using the trained model.
        :param image_path: Path to the input image.
        :return: Predicted class label and probability.
        """
        # Load and preprocess the image
        img = load_img(image_path, target_size=(self.img_height, self.img_width))
        img_array = img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict
        prediction = self.model.predict(img_array)
        class_label = "Original" if prediction[0][0] > 0.5 else "GenAI"
        probability = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

        return class_label, probability

# Example Usage
if __name__ == "__main__":
    classifier = ImageAuthenticityClassifier(img_height=128, img_width=128, batch_size=32, epochs=20)

    # Directories for dataset
    train_dir = "data/train"
    val_dir = "data/validate"
    test_dir = "data/test"

    # Prepare the dataset
    classifier.prepare_dataset(train_dir, val_dir, test_dir)

    # Build the model
    classifier.build()

    # Train the model
    classifier.train()

    # Save the model
    classifier.save_model("genai_vs_original_model.keras")

    # Plot training loss and accuracy
    classifier.plot_loss()
    classifier.plot_accuracy()

    # Evaluate the model on the test dataset
    classifier.evaluate()
