import os
import torch
import argparse
import pyaudio
import wave
from zipfile import ZipFile
import langid
import se_extractor
from api import BaseSpeakerTTS, ToneColorConverter
import openai
import logging
import re
import numpy as np
import time
import threading
from faster_whisper import WhisperModel
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr  # <-- Add this line
import requests
import wikipedia

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RED = '\033[91m'
RESET_COLOR = '\033[0m'

style = 'default'  # or 'style', depending on your preference

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
args = parser.parse_args()

# Model and device setup
en_ckpt_base = 'checkpoints/base_speakers/EN'
ckpt_converter = 'checkpoints/converter'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'outputs'
output_dir1 = 'outputs1'
os.makedirs(output_dir, exist_ok=True)

# Load models
en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device="cuda" if torch.cuda.is_available() else "cpu")
en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device="cuda" if torch.cuda.is_available() else "cpu")
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# Load speaker embeddings for English
en_source_default_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
en_source_style_se = torch.load(f'{en_ckpt_base}/en_style_se.pth').to(device)


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global flags and threads
conversation_thread = None
stop_audio_flag = False
stop_generation_flag = False

# Function to perform internet search and return results
def search_online(query):
    try:
        # You can replace this with any search API, for example, Wikipedia
        search_results = wikipedia.summary(query, sentences=50)  # Get the first 50 sentences from Wikipedia

        logger.info(f"Search results for '{query}': {search_results}")
        return search_results

    except wikipedia.exceptions.DisambiguationError as e:
        logger.warning(f"Disambiguation error for query '{query}': {e.options}")
        return "Multiple results found. Please be more specific."
    except wikipedia.exceptions.HTTPTimeoutError:
        logger.error(f"Timeout error while searching for '{query}'")
        return "The search request timed out."
    except Exception as e:
        logger.error(f"Error during online search: {e}")
        return "An error occurred while searching online."

# Function to play a beep sound (simple tone)
def play_beep(frequency=1000, duration=0.3):
    p = pyaudio.PyAudio()
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, output=True)
    stream.write(audio_data.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()


# Function to clean the transcription
def clean_transcription(transcription):
    transcription = re.sub(r'[\u0300-\u036f]', '', transcription)  # Remove diacritics
    transcription = ''.join([char for char in transcription if ord(char) < 128]).strip()
    transcription = re.sub(r'length:\d+', '', transcription)
    return transcription


# Function to initialize models
def initialize_models():
    try:
        whisper_model = WhisperModel("base", device="cuda" if torch.cuda.is_available() else "cpu")
        openai_client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
        tts_model = BaseSpeakerTTS('checkpoints/base_speakers/EN/config.json', device='cuda' if torch.cuda.is_available() else 'cpu')
        tts_model.load_ckpt('checkpoints/base_speakers/EN/checkpoint.pth')
        return whisper_model, openai_client, tts_model
    except Exception as e:
        logger.error(f"Error during model initialization: {e}")
        return None, None, None


# Audio Playing Function
def play_audio(file_path):
    logger.info(f"Playing audio: {file_path}")
    wf = wave.open(file_path, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)
    stream.stop_stream()
    stream.close()
    p.terminate()


# Record Audio Function
def record_audio(file_path, silence_threshold=500, silence_duration=2):
    logger.info(f"Recording... Will stop after {silence_duration} seconds of silence.")
    play_beep(frequency=1000, duration=0.3)

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

    frames = []
    silent_frames = 0

    while True:
        data = stream.read(1024)
        frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(np.square(audio_data)))
        if rms < silence_threshold:
            silent_frames += 1
        else:
            silent_frames = 0
        if silent_frames > silence_duration * 16000 / 1024:
            logger.info(f"Silence detected for {silence_duration} seconds. Stopping recording.")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()
    play_beep(frequency=500, duration=0.3)
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()


# Transcribe audio with Whisper
def transcribe_with_whisper(whisper_model, audio_file_path):
    segments, _ = whisper_model.transcribe(audio_file_path, language="en", without_timestamps=True)
    transcription = "".join(segment.text for segment in segments)
    return clean_transcription(transcription)


# Streaming LLM with TTS playback (process tokens and accumulate sentences)
# Modify llm_streamed to print colored logs
def llm_streamed(openai_client, user_input, system_message, conversation_history, retrieved_data, tts_model):
    # Prepare the initial system message and user input for the conversation history
    messages = [{"role": "system", "content": system_message}] + conversation_history + [
        {"role": "user", "content": user_input},
        {"role": "system", "content": "Here is the relevant information I found: " + retrieved_data}]

    logger.info(f"{NEON_GREEN}Sending user input to LLM: {RED}{user_input}{RESET_COLOR}")  # User input in red

    # Streaming completion call with real-time processing
    streamed_completion = openai_client.chat.completions.create(
        model="local-model",
        messages=messages,
        stream=True  # Ensure this is set to stream tokens
    )

    response = ""
    sentence_buffer = ""  # Buffer to accumulate tokens for a full sentence
    save_path = 'outputs/output_partial.wav'

    # Process each chunk/token as it arrives
    for chunk in streamed_completion:
        content = chunk.choices[0].delta.content
        if content:
            response += content
            sentence_buffer += content  # Accumulate tokens into the sentence buffer
            logger.info(f"{NEON_GREEN}Received token: {content}{RESET_COLOR}")  # Tokens in green

            # Check if sentence-ending punctuation exists
            if sentence_buffer.endswith(('.', '!', '?')):
                tts_model = en_base_speaker_tts
                source_se = en_source_default_se if style == 'default' else en_source_style_se

                target_se, audio_name = se_extractor.get_se(r"C:\Users\ameya\Downloads\lisa_voice_sample.mp3", tone_color_converter, target_dir='processed',
                                                            vad=True)

                src_path = f'{output_dir}/tmp.wav'
                tts_model.tts(sentence_buffer.strip(), src_path, speaker="default", language='English')


                save_path = f'{output_dir}/output.wav'
                save_path1 = f'{output_dir1}/output.wav'

                # Run the tone color converter
                encode_message = "@MyShell"
                tone_color_converter.convert(audio_src_path=src_path, src_se=source_se, tgt_se=target_se,
                                             output_path=save_path, message=encode_message)

                # Run the tone color converter
                encode_message = "@MyShell"
                tone_color_converter.convert(audio_src_path=src_path, src_se=source_se, tgt_se=target_se,
                                             output_path=save_path1, message=encode_message)

                # Send the complete sentence to TTS and play it
                logger.info(f"{GREEN}Sending full sentence to TTS: {RESET_COLOR}{BLUE}{sentence_buffer.strip()}{RESET_COLOR}")
                play_audio(save_path)  # Play the generated TTS audio
                os.remove(save_path)
                sentence_buffer = ""  # Clear the buffer for the next sentence

    return clean_transcription(response)


# Retrieve relevant data
def retrieve_relevant_data(user_input, document_content):
    chunks = [document_content[i:i + 500] for i in range(0, len(document_content), 500)]
    vectorizer = TfidfVectorizer(stop_words="english")
    knowledge_vectors = vectorizer.fit_transform(chunks)
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, knowledge_vectors)
    most_similar_idx = similarities.argmax()
    return chunks[most_similar_idx]


# Add text to knowledge base
def add_to_knowledge_base(text, document_path="knowledge_base.txt"):
    with open(document_path, 'a', encoding='utf-8') as file:
        file.write("\n" + text)
    logger.info(f"Text added to knowledge base: {text}")


# Stop conversation and model generation
def stop_conversation():
    global stop_audio_flag, stop_generation_flag
    stop_audio_flag = True
    stop_generation_flag = True
    logger.info("Stopping conversation thread...")


# Modify the user_chatbot_conversation function to include search functionality
def user_chatbot_conversation():
    global conversation_thread
    while True:
        try:
            whisper_model, openai_client, tts_model = initialize_models()
            if not whisper_model or not openai_client or not tts_model:
                raise Exception("Failed to initialize models")

            conversation_history = []
            system_message = "You are Lisa from BlackPink, my girlfriend who loves me so much. Be romantic and Talk in her style"

            document_path = "knowledge_base.txt"
            with open(document_path, 'r', encoding='utf-8') as file:
                document_content = file.read()

            while True:
                audio_file = "temp_recording.wav"
                record_audio(audio_file, silence_threshold=500, silence_duration=2)

                try:
                    user_input = transcribe_with_whisper(whisper_model, audio_file)
                    os.remove(audio_file)
                except Exception as e:
                    logger.error(f"Error in transcription: {e}")
                    continue

                if "exit conversation" in user_input.lower():
                    logger.info("Exit command detected. Returning to wake word listening...")
                    return  # Exit the function but go back to listen_for_wake_word()

                if user_input.lower().startswith("remember"):
                    remember_text = user_input[8:].strip()
                    add_to_knowledge_base(remember_text)
                    continue

                if user_input.lower().startswith("search for"):
                    query = user_input[11:].strip()
                    logger.info(f"Search query: {query}")

                    # Perform the online search and get the results
                    search_results = search_online(query)

                    # Append the search results to the knowledge base
                    add_to_knowledge_base(search_results, document_path)
                    continue

                retrieved_data = retrieve_relevant_data(user_input, document_content)
                chatbot_response = llm_streamed(openai_client, user_input, system_message, conversation_history,
                                                    retrieved_data, tts_model)
                conversation_history.append({"role": "assistant", "content": chatbot_response})

                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-20:]

        except KeyboardInterrupt:
            logger.info("Ctrl+C detected. Restarting the conversation...")
            continue  # Restart the outer loop for a fresh conversation
        except Exception as e:
            logger.error(f"Error in conversation: {e}")
            break

# Wake word detection using offline pocketsphinx
#def listen_for_wake_word():
#    recognizer = sr.Recognizer()
#    mic = sr.Microphone()

#    with mic as source:
#        logger.info("Listening for wake word...")
#        recognizer.adjust_for_ambient_noise(source)
#        while True:
#            audio = recognizer.listen(source)
#            try:
#                # Use pocketsphinx for offline speech recognition
#                text = recognizer.recognize_sphinx(audio).lower()
#                if "computer" in text:
#                    logger.info("Wake word detected! Starting conversation.")
#                    user_chatbot_conversation()
#            except sr.UnknownValueError:
#                pass  # No speech recognized
#            except sr.RequestError as e:
#                logger.error(f"Error with the speech recognition service: {e}")
#                break


# Wake word detection
def listen_for_wake_word():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        logger.info("Listening for wake word...")
        recognizer.adjust_for_ambient_noise(source)
        while True:
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio).lower()
                if "hey lisa" in text:
                    logger.info("Wake word detected! Starting conversation.")
                    user_chatbot_conversation()
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                logger.error(f"Error with the speech recognition service: {e}")
                break


# Safe start of conversation thread
def start_conversation():
    global conversation_thread
    if conversation_thread is None or not conversation_thread.is_alive():
        conversation_thread = threading.Thread(target=user_chatbot_conversation)
        conversation_thread.start()
    else:
        logger.info("Conversation thread is already running.")


if __name__ == "__main__":
    try:
        while True:  # Infinite loop to keep the program running
            listen_for_wake_word()  # Start listening for the wake word
    except KeyboardInterrupt:
        logger.info("Program interrupted. Exiting gracefully.")

