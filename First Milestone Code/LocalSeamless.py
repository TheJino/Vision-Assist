import cv2
import torch
import time
import threading
import os
import pygame
import pyttsx3  # ✅ Using pyttsx3 for offline TTS
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# -----------------------------
# 1. Load Image Captioning Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "nlpconnect/vit-gpt2-image-captioning"

model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)  # Updated from ViTFeatureExtractor
tokenizer = AutoTokenizer.from_pretrained(model_name)

# -----------------------------
# 2. Initialize `pyttsx3` for Offline Text-to-Speech
# -----------------------------
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Adjust speech speed
engine.setProperty("volume", 1.0)  # Max volume

# -----------------------------
# 3. Function to Generate Captions
# -----------------------------
def generate_caption(image):
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    outputs = model.generate(inputs["pixel_values"], max_length=16, num_beams=4)
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption

# -----------------------------
# 4. Function to Convert Text to Speech Using `pyttsx3`
# -----------------------------
def text_to_speech(text):
    print(f"🔊 Speaking: {text}")
    engine.say(text)
    engine.runAndWait()  # Blocks until speech is finished

# -----------------------------
# 5. Function to Handle Automatic Photo Capture Every 10 Seconds
# -----------------------------
def capture_and_process():
    global last_capture_time, cap
    capture_interval = 10  # Capture an image every 10 seconds

    while True:
        current_time = time.time()
        remaining_time = capture_interval - (current_time - last_capture_time)

        if remaining_time > 0:
            print(f"⏳ Next capture in {int(remaining_time)} seconds...")
            time.sleep(remaining_time)  # Wait until next capture

        last_capture_time = time.time()  # Reset cooldown timer
        print("📸 Capturing a photo...")

        # Ensure the camera is still open
        if not cap.isOpened():
            print("⚠️ Camera was closed, reinitializing...")
            cap = initialize_camera()

        ret, frame = cap.read()

        if not ret:
            print("❌ Failed to capture frame. Retrying...")
            time.sleep(2)  # Wait a bit before retrying
            continue

        # Resize frame for processing
        resized_frame = cv2.resize(frame, (224, 224))

        # Generate caption
        caption = generate_caption(resized_frame)
        print("📝 Caption:", caption)

        # Provide audio feedback using `pyttsx3`
        text_to_speech(caption)

# -----------------------------
# 6. Improved Camera Initialization with Retries
# -----------------------------
def initialize_camera(retry_limit=10, delay_between_retries=2):
    """Tries to initialize the camera multiple times before giving up."""
    for attempt in range(1, retry_limit + 1):
        print(f"Attempt {attempt}/{retry_limit} to initialize the camera...")

        cap = cv2.VideoCapture(0)  # Open the camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(2)  # Allow the camera to warm up

        if cap.isOpened():
            print("✅ Camera successfully initialized!")
            return cap  # Return the working camera object
        
        print("❌ Camera not detected, retrying in 2 seconds...")
        time.sleep(delay_between_retries)

    print("⛔ Failed to initialize camera after multiple attempts.")
    exit()

# -----------------------------
# 7. Start the Live Feed
# -----------------------------
cap = initialize_camera()
last_capture_time = time.time()  # Start from current time

# Start the capture thread
thread = threading.Thread(target=capture_and_process, daemon=True)
thread.start()

print("📷 Live camera feed is running. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame. Retrying in 2 seconds...")
            time.sleep(2)
            continue  # Retry frame capture

        cv2.imshow("Camera Feed", frame)

        # Exit the program if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nExiting...")

# Cleanup
cap.release()
cv2.destroyAllWindows()
