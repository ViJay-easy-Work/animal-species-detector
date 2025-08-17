import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
import cv2
from PIL import Image, ImageTk
import time
import psutil
import tensorflow as tf
import numpy as np

# Load your model (replace with actual model path if saved)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Function to preprocess images for the model
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    return image / 255.0

# List of general animal names to identify
general_animals = [
    "dog", "cat", "bird", "fish", "horse", "cow", "elephant", "tiger", "lion", "monkey",
    "deer", "rabbit", "bear", "wolf", "leopard", "zebra", "kangaroo", "giraffe", "camel",
    "sheep", "goat", "pig", "duck", "chicken", "frog", "turtle", "snake", "parrot", "eagle",
    "owl", "pigeon", "sparrow", "hamster", "mouse", "rat", "antelope", "buffalo", "cheetah",
    "chimpanzee", "crocodile", "dolphin", "fox", "hawk", "hippopotamus", "jaguar", "lobster",
    "octopus", "panther", "penguin", "whale"
]

# Initialize GUI window
root = tk.Tk()
root.title("Animal Species Detection")
root.geometry("800x600")

# Function to update time and CPU usage
def update_time_cpu():
    cpu_usage = psutil.cpu_percent()
    time_str = time.strftime('%Y-%m-%d %H:%M:%S')
    time_label.config(text=f"Time: {time_str}")
    cpu_label.config(text=f"CPU Usage: {cpu_usage}%")
    root.after(1000, update_time_cpu)

# Function to display an image in the GUI
def display_image(image):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = ImageTk.PhotoImage(image)
    image_label.config(image=image)
    image_label.image = image

# Function to start detection with the camera
def start_detection():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        display_image(frame)
        detect_species(frame)

# Function to upload and detect species from an image
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        display_image(image)
        detect_species(image)

# Function to detect general animal type and species
def detect_species(image):
    preprocessed = preprocess_image(image)
    predictions = model.predict(preprocessed)
    
    # Decode predictions to get top results
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    
    # Identify the general animal type from the predictions
    animal_name = None
    species_name = None
    confidence = 0
    
    for _, label, score in decoded:
        if any(animal in label for animal in general_animals):
            animal_name = label
            confidence = score
            break  # Stop once the first relevant animal type is detected
    
    # Display general animal type in GUI
    if animal_name:
        animal_label.config(text=f"Animal Detected: {animal_name}")
        progress_bar['value'] = confidence * 100
        unsure_progress['value'] = (1 - confidence) * 100
        
        # Assign species name if it differs from general animal type
        for _, label, score in decoded:
            if label != animal_name:
                species_name = label
                break
        species_label.config(text=f"Species: {species_name if species_name else 'N/A'}")
    
    else:
        # If no general animal detected, show highest confidence prediction
        top_prediction = decoded[0]
        animal_label.config(text=f"Detected: {top_prediction[1]}")
        species_label.config(text="Species: N/A")
        progress_bar['value'] = top_prediction[2] * 100
        unsure_progress['value'] = (1 - top_prediction[2]) * 100



# Function to save the image and result
def save_screenshot():
    file_path = filedialog.asksaveasfilename(defaultextension=".png")
    if file_path:
        if image_label.image:
            image_label.image.save(file_path)
            messagebox.showinfo("Success", "Screenshot saved successfully!")

# UI Components
start_button = tk.Button(root, text="Start Detection", command=start_detection)
start_button.pack()

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack()

save_button = tk.Button(root, text="Save Screenshot", command=save_screenshot)
save_button.pack()

image_label = tk.Label(root)
image_label.pack()

animal_label = tk.Label(root, text="Animal Detected: N/A", font=("Helvetica", 14))
animal_label.pack()

species_label = tk.Label(root, text="Species: N/A", font=("Helvetica", 14))
species_label.pack()

confidence_label = tk.Label(root, text="Confidence: N/A")
confidence_label.pack()

progress_bar = Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress_bar.pack()

unsure_progress = Progressbar(root, orient="horizontal", length=300, mode="determinate")
unsure_progress.pack()

time_label = tk.Label(root, text="Time: N/A")
time_label.pack()

cpu_label = tk.Label(root, text="CPU Usage: N/A")
cpu_label.pack()

# Start updating time and CPU usage
update_time_cpu()

root.mainloop()
