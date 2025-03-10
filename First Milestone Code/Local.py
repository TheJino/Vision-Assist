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

# Set voice properties (optional)
engine.setProperty("rate", 150)  # Adjust speech speed (default ~200)
engine.setProperty("volume", 1.0)  # Adjust volume (0.0 to 1.0)

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
# 5. Function to Handle Photo Capture with a Cooldown
# -----------------------------
def capture_and_process(cap):
    global last_capture_time
    while True:
        current_time = time.time()
        if current_time - last_capture_time < 20:
            remaining_time = int(20 - (current_time - last_capture_time))
            print(f"Please wait {remaining_time} more seconds before capturing again.")
            time.sleep(remaining_time)

        input("Press Enter to capture a photo...")

        last_capture_time = time.time()  # Reset cooldown timer
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            continue

        # Resize frame for processing
        resized_frame = cv2.resize(frame, (224, 224))

        # Generate caption
        caption = generate_caption(resized_frame)
        print("Caption:", caption)

        # Provide audio feedback using `pyttsx3`
        text_to_speech(caption)

# -----------------------------
# 6. Improved Camera Initialization with Retries
# -----------------------------
def initialize_camera(retry_limit=10, delay_between_retries=2):
    """Tries to initialize the camera multiple times before giving up."""
    cap = None
    for attempt in range(1, retry_limit + 1):
        print(f"Attempt {attempt}/{retry_limit} to initialize the camera...")

        cap = cv2.VideoCapture(0)
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

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize cooldown timer
last_capture_time = 0

# Start the capture thread
thread = threading.Thread(target=capture_and_process, args=(cap,))
thread.daemon = True
thread.start()

print("📷 Live camera feed is running. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        cv2.imshow("Camera Feed", frame)

        # Exit the program if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nExiting...")

# Cleanup
cap.release()
cv2.destroyAllWindows()
