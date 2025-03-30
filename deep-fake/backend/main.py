import os
import numpy as np
import random
import base64
import io

# -------------------------
# TensorFlow / Keras
# -------------------------
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Embedding,
                                     LSTM, Flatten, Concatenate, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------
# FastAPI & Utilities
# -------------------------
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import uvicorn

# -------------------------
# Visualization (optional)
# -------------------------
import matplotlib.pyplot as plt

##############################################
# GAN CLASS (Simplified for Synthetic Comments)
##############################################
class TextGAN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, lstm_units):
        super(TextGAN, self).__init__()
        self.generator = tf.keras.Sequential([
            Embedding(vocab_size, embedding_dim),
            LSTM(lstm_units, return_sequences=True),
            LSTM(lstm_units),
            Dense(vocab_size, activation='softmax')
        ])

    def call(self, noise):
        return self.generator(noise)


def generate_synthetic_comments(tokenizer, num_samples=500, max_length=50):
    """
    Generate random sequences (as a placeholder for a real GAN).
    Then convert them to text via the tokenizer.
    """
    vocab_size = len(tokenizer.word_index) + 1
    text_gan = TextGAN(vocab_size, 128, 64)
    # We're not really training the GAN here, just generating random sequences.
    synthetic_sequences = np.random.randint(1, vocab_size, size=(num_samples, max_length))
    synthetic_comments = tokenizer.sequences_to_texts(synthetic_sequences)
    return synthetic_comments


#############################
#  TRAINING LOGIC
#############################
def train_and_save_model():
    """
    Trains the multimodal model (image + text), saves the final model.
    You can comment out this function if you only want to serve the model.
    """

    # -----------------------
    # Step 1: Text Data (and random "GAN" generation)
    # -----------------------
    text_samples = [
        "Fake image detected",
        "Real person image",
        "AI-generated content",
        "Authentic photograph"
    ]
    num_words = 5000
    max_length = 50
    tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(text_samples)

    # Generate synthetic comments
    synthetic_comments = generate_synthetic_comments(tokenizer, num_samples=1000, max_length=max_length)
    all_comments = text_samples + synthetic_comments
    sequences = tokenizer.texts_to_sequences(all_comments)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    # -----------------------
    # Step 2: Image Data
    # -----------------------
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 5
    EPOCHS = 2  # set small for demo; increase for real training
    LEARNING_RATE = 0.0001

    BASE_DIR = "/Users/prakharsingh/Desktop/Learning/project/deep-fake/backend/Dataset"  # Change to your dataset path
    TRAIN_DIR = os.path.join(BASE_DIR, 'Train')
    VALID_DIR = os.path.join(BASE_DIR, 'Validation')
    TEST_DIR = os.path.join(BASE_DIR, 'Test')

    train_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
    valid_generator = valid_datagen.flow_from_directory(
        VALID_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

    # -----------------------
    # Step 3: Build Multi-Modal Model
    # -----------------------
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)

    # Text branch
    text_input = Input(shape=(max_length,))
    text_embedding = Embedding(input_dim=num_words, output_dim=128)(text_input)
    text_lstm = LSTM(64)(text_embedding)

    merged = Concatenate()([x, text_lstm])
    out = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[base_model.input, text_input], outputs=out)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])

    # -----------------------
    # Step 4: Training
    # -----------------------
    num_train_samples = train_generator.samples
    num_valid_samples = valid_generator.samples

    # Randomly sample text sequences to match the number of images in the dataset
    random_texts_train = np.random.choice(padded_sequences.shape[0], num_train_samples, replace=True)
    text_train_data = padded_sequences[random_texts_train]

    random_texts_valid = np.random.choice(padded_sequences.shape[0], num_valid_samples, replace=True)
    text_valid_data = padded_sequences[random_texts_valid]

    def data_generator(image_gen, text_data):
        idx = 0
        while True:
            img_batch, labels = next(image_gen)
            batch_size = len(img_batch)
            # text_data must align with the same batch size
            text_batch = text_data[idx: idx + batch_size]
            if len(text_batch) < batch_size:
                # wrap around
                idx = 0
                text_batch = text_data[idx: idx + batch_size]
            idx += batch_size
            yield ((img_batch, text_batch), labels)

            import tensorflow as tf

    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(train_generator, text_train_data),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),  # image batch
                tf.TensorSpec(shape=(None, 50), dtype=tf.int32),             # text batch
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)  # labels
        )
    )

    valid_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(valid_generator, text_valid_data),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 50), dtype=tf.int32),
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )


    train_data_gen = data_generator(train_generator, text_train_data)
    valid_data_gen = data_generator(valid_generator, text_valid_data)

    # Train the model
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=valid_dataset,
        steps_per_epoch=len(train_generator),
        validation_steps=len(valid_generator)
    )


    # Save final model
    model.save("deepfake_multimodal_model.h5")
    print("Model saved as 'deepfake_multimodal_model.h5'")

    # Test evaluation
    random_texts_test = np.random.choice(padded_sequences.shape[0], test_generator.samples, replace=True)
    text_test_data = padded_sequences[random_texts_test]

    def test_data_generator(image_gen, text_data):
        idx = 0
        while True:
            img_batch, labels = next(image_gen)
            batch_size = len(img_batch)
            text_batch = text_data[idx: idx + batch_size]
            if len(text_batch) < batch_size:
                idx = 0
                text_batch = text_data[idx: idx + batch_size]
            idx += batch_size
            yield ((img_batch, text_batch), labels) 

    test_dataset = tf.data.Dataset.from_generator(
        lambda: test_data_generator(test_generator, text_valid_data),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 50), dtype=tf.int32),
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )

    test_loss, test_accuracy = model.evaluate(
        test_dataset,
        steps=len(test_generator)
    )
    print(f"Test Accuracy: {test_accuracy:.4f}")


#########################################
# FASTAPI APP FOR INFERENCE
#########################################

# We’ll create an app that loads the saved model and
# provides endpoints for inference (and optional text generation).
app = FastAPI(title="Deepfake Multimodal API", version="1.0.0")

# Add this block:
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev, you can keep "*" or specify ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global variables for model, tokenizer, etc.
LOADED_MODEL = None
TOKENIZER = None
MAX_LENGTH = 50
NUM_WORDS = 5000

# ---- Pydantic Models for request/response ----
class PredictRequest(BaseModel):
    text_input: Optional[str] = None
    image_b64: Optional[str] = None  # if user sends image as base64


class PredictResponse(BaseModel):
    prediction: str  # "Fake" or "Real"
    confidence: float


class GenerateCommentRequest(BaseModel):
    prompt: Optional[str] = None


class GenerateCommentResponse(BaseModel):
    generated_text: str


@app.on_event("startup")
def load_resources():
    """
    Load your saved model and any other assets (tokenizer, etc.) at startup.
    """
    global LOADED_MODEL, TOKENIZER

    # 1) Load saved Keras model
    print("Loading model from 'deepfake_multimodal_model.h5' ...")
    LOADED_MODEL = tf.keras.models.load_model("deepfake_multimodal_model.h5")

    # 2) Re-create tokenizer if needed
    #    We know from training we used num_words=5000, oov_token='<OOV>'
    print("Re-initializing Tokenizer ...")
    TOKENIZER = Tokenizer(num_words=NUM_WORDS, oov_token='<OOV>')
    # For a real project, you’d fit the tokenizer on the same corpus used in training,
    # then save & load the tokenizer. Here, we’ll just do a trivial fit:
    dummy_samples = [
        "Fake image detected",
        "Real person image",
        "AI-generated content",
        "Authentic photograph"
    ]
    TOKENIZER.fit_on_texts(dummy_samples)  # minimal fitting for demonstration

    print("Resources loaded.")


def preprocess_image_from_base64(image_b64: str):
    """
    Decode a base64-encoded image and resize/preprocess for MobileNetV2 input.
    """
    image_data = base64.b64decode(image_b64)
    pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
    pil_image = pil_image.resize((224, 224))
    img_array = np.array(pil_image) / 255.0
    # Add batch dimension
    return np.expand_dims(img_array, axis=0)


def preprocess_text(text: str):
    """
    Convert text to padded sequence for the LSTM branch.
    """
    seq = TOKENIZER.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post')
    return padded


@app.get("/")
def root():
    return {"message": "Welcome to the Deepfake Multimodal API"}


@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    """
    Endpoint to classify an image+text pair as "Fake" or "Real".
    - text_input: optional text
    - image_b64: optional image in Base64
    """
    if not data.text_input and not data.image_b64:
        return {"prediction": "Error", "confidence": 0.0}

    # Preprocess text
    if data.text_input:
        text_array = preprocess_text(data.text_input)
    else:
        # If no text, we can pass zeros. Make sure shape matches (1, MAX_LENGTH).
        text_array = np.zeros((1, MAX_LENGTH))

    # Preprocess image
    if data.image_b64:
        image_array = preprocess_image_from_base64(data.image_b64)
    else:
        # If no image, pass a dummy array shaped (1,224,224,3).
        image_array = np.zeros((1, 224, 224, 3))

    # Predict
    preds = LOADED_MODEL.predict([image_array, text_array])[0][0]
    # preds is a float in [0,1]. Let's say >0.5 => Fake
    label = "Fake" if preds > 0.5 else "Real"
    confidence = float(preds) if preds > 0.5 else 1 - float(preds)

    return PredictResponse(prediction=label, confidence=confidence)


@app.post("/generate_comment", response_model=GenerateCommentResponse)
def generate_comment_endpoint(req: GenerateCommentRequest):
    """
    (Optional) Endpoint to demonstrate text generation (placeholder).
    In practice, you'd load your real generator or at least a random snippet.
    """
    # For demo, we just output random text from the tokenizer word_index:
    word_list = list(TOKENIZER.word_index.keys())
    random_text = " ".join(random.choices(word_list, k=10))
    return GenerateCommentResponse(generated_text=random_text)


# ---------------------------------------------
# MAIN ENTRY POINT (run training or serve)
# ---------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", 
        default="serve", 
        help="Choose 'train' to train the model, or 'serve' to run the FastAPI server."
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_and_save_model()
    else:
        # Run the API server
        uvicorn.run(app, port=8000)
