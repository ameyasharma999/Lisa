import bpy
import socket
import threading
import os
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to list all available shape keys for the selected object
def get_available_shape_keys(obj):
    if obj.data.shape_keys:
        return [key.name for key in obj.data.shape_keys.key_blocks]
    return []

# Function to dynamically map phonemes to available shape keys
def get_phoneme_to_shape_mapping():
    phoneme_to_shape = {
        # Consonants
        "p": "Mouth_Pucker", "b": "Mouth_Pucker", "m": "Mouth_Pucker",
        "f": "Mouth_Smile", "v": "Mouth_Smile", "θ": "Mouth_Smile", "ð": "Mouth_Smile",
        "t": "Mouth_Frown", "d": "Mouth_Frown", "n": "Mouth_Frown", "l": "Mouth_Smile",
        "r": "Mouth_Smile", "s": "Mouth_Frown", "z": "Mouth_Frown", "ʃ": "Mouth_Frown",
        "ʒ": "Mouth_Frown", "tʃ": "Mouth_Frown", "dʒ": "Mouth_Frown", "k": "Mouth_Frown",
        "g": "Mouth_Frown", "w": "Mouth_Round", "j": "Mouth_Smile", "h": "Mouth_Lips_Part",
        "ŋ": "Mouth_Frown", "ʍ": "Mouth_Round", "ɹ": "Mouth_Round", "ɫ": "Mouth_Frown",  # Added ɫ
        "ʌ": "Mouth_Open", "ɑ": "Mouth_Open",  # New vowels added
        # Vowels
        "i": "Mouth_Smile", "ɪ": "Mouth_Smile", "e": "Mouth_Open", "ɛ": "Mouth_Open",
        "æ": "Mouth_Open", "a": "Mouth_Open", "ʌ": "Mouth_Open", "ɑ": "Mouth_Open",
        "ɔ": "Mouth_Open", "o": "Mouth_Round", "ʊ": "Mouth_Pucker", "u": "Mouth_Round",  # Added u, o
        "ə": "Mouth_Lips_Part", "ɜ": "Mouth_Open", "ɔɪ": "Mouth_Smile",
        "aɪ": "Mouth_Open", "aʊ": "Mouth_Round", "eɪ": "Mouth_Open", "oʊ": "Mouth_Round",
        "iː": "Mouth_Smile", "uː": "Mouth_Round",
    }
    # Get the available shape keys from the object
    obj_name = "CC_Base_Body"  # Replace with the actual object name
    obj = bpy.data.objects.get(obj_name)

    if obj:
        available_keys = get_available_shape_keys(obj)
        # Filter the phoneme-to-shape mapping to include only available shape keys
        logging.info(f"Available shape keys: {available_keys}")
        return {phoneme: shape for phoneme, shape in phoneme_to_shape.items() if shape in available_keys}
    
    logging.error(f"Object '{obj_name}' not found.")
    return {}

# Function to set shape key values with keyframe insertion
def set_shape_key(phoneme, value, frame):
    phoneme_to_shape = get_phoneme_to_shape_mapping()

    shape_key_name = phoneme_to_shape.get(phoneme, "Mouth_Smile")  # Default to Mouth_Smile if phoneme not found
    obj_name = "CC_Base_Body"  # Replace with the actual object name
    obj = bpy.data.objects.get(obj_name)

    if not obj:
        logging.error(f"Object '{obj_name}' not found.")
        return

    if obj.data.shape_keys:
        shape_key = obj.data.shape_keys.key_blocks.get(shape_key_name)
        if shape_key:
            shape_key.value = value
            shape_key.keyframe_insert(data_path="value", frame=frame)  # Insert keyframe at the given frame
            logging.info(f"Applied shape key '{shape_key_name}' with value {value} at frame {frame}.")
        else:
            logging.error(f"Shape key '{shape_key_name}' not found in object '{obj_name}'.")
    else:
        logging.error(f"Object '{obj_name}' has no shape keys.")

# Function to reset individual shape keys to 0 after the phoneme duration
def reset_shape_key(phoneme, frame):
    phoneme_to_shape = get_phoneme_to_shape_mapping()

    shape_key_name = phoneme_to_shape.get(phoneme, "Mouth_Smile")  # Default to Mouth_Smile if phoneme not found
    obj_name = "CC_Base_Body"  # Replace with the actual object name
    obj = bpy.data.objects.get(obj_name)

    if not obj:
        logging.error(f"Object '{obj_name}' not found.")
        return

    if obj.data.shape_keys:
        shape_key = obj.data.shape_keys.key_blocks.get(shape_key_name)
        if shape_key:
            shape_key.value = 0.0  # Reset shape key value
            shape_key.keyframe_insert(data_path="value", frame=frame)  # Insert keyframe at the reset frame
            logging.info(f"Reset shape key '{shape_key_name}' to 0.0 at frame {frame}.")
        else:
            logging.error(f"Shape key '{shape_key_name}' not found in object '{obj_name}'.")
    else:
        logging.error(f"Object '{obj_name}' has no shape keys.")

# Function to reset all shape keys to 0 (used if needed)
def reset_all_shape_keys():
    obj_name = "CC_Base_Body"  # Replace with the actual object name
    obj = bpy.data.objects.get(obj_name)

    if obj and obj.data.shape_keys:
        for shape_key in obj.data.shape_keys.key_blocks:
            shape_key.value = 0.0
        logging.info("Reset all shape keys to 0.0.")

# Main function to handle incoming phonemes and apply shape key adjustments
def handle_phonemes():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', 65433))  # Ensure this matches the client
    server.listen(1)  # Listen for incoming connections
    logging.info("Server listening for phonemes...")

    frame = 1  # Initial frame for the first phoneme
    phoneme_duration = 5000000000000000  # Duration (in frames) each phoneme should last

    while True:
        try:
            conn, addr = server.accept()  # Accept a new connection
            logging.info(f"Connected to {addr}")
            with conn:
                while True:
                    data = conn.recv(1024)  # Receive phoneme data
                    if not data:
                        break  # Break if connection is closed
                    phonemes = data.decode('utf-8').split()  # Decode and split phonemes
                    logging.info(f"Received phonemes: {phonemes}")

                    for phoneme in phonemes:
                        # Apply the shape key for the current phoneme
                        set_shape_key(phoneme, 1.0, frame)
                        frame += phoneme_duration  # Wait for the phoneme duration

                        # Reset the shape key after the duration
                        reset_shape_key(phoneme, frame)
                        frame += phoneme_duration  # Move to the next frame

                        logging.info(f"Processed phoneme '{phoneme}' sequentially.")
        except Exception as e:
            logging.error(f"Error occurred: {e}")

# Start a new thread for the phoneme handling function
def start_phoneme_server():
    threading.Thread(target=handle_phonemes, daemon=True).start()

# Call the function to start the server
start_phoneme_server()
