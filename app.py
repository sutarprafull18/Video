import streamlit as st
import os
import tempfile
import logging
from pathlib import Path

# First, let's check and install required packages if they're missing
def install_missing_packages():
    import subprocess
    import sys
    
    packages = {
        'moviepy': 'moviepy',
        'whisper': 'openai-whisper',
        'torch': 'torch torchvision torchaudio',
        'googletrans': 'googletrans==3.1.0a0',
        'gtts': 'gTTS'
    }
    
    for package, pip_name in packages.items():
        try:
            __import__(package)
        except ImportError:
            st.warning(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            st.success(f"Installed {package}")

# Install missing packages
install_missing_packages()

# Now import the required packages
import whisper
import torch
from googletrans import Translator
from moviepy.editor import VideoFileClip, AudioFileClip
from gtts import gTTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing VideoProcessor with device: {self.device}")
        self.model = whisper.load_model("base", device=self.device)
        self.translator = Translator()

    def transcribe_audio(self, audio_path):
        """Transcribe audio to text using Whisper"""
        try:
            result = self.model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise

    def translate_text(self, text, target_lang):
        """Translate text to target language"""
        try:
            translation = self.translator.translate(text, dest=target_lang)
            return translation.text
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            raise

    def create_audio(self, text, lang, output_path):
        """Create audio from text using gTTS"""
        try:
            tts = gTTS(text=text, lang=lang)
            tts.save(output_path)
        except Exception as e:
            logger.error(f"Text-to-speech error: {str(e)}")
            raise

def create_temp_dir():
    """Create a temporary directory and return its path"""
    temp_dir = tempfile.mkdtemp()
    return temp_dir

def process_video(video_file, progress_placeholder, status_placeholder):
    """Process a single video file"""
    temp_dir = create_temp_dir()
    
    try:
        # Save uploaded video to temp directory
        input_path = os.path.join(temp_dir, "input_video.mp4")
        with open(input_path, "wb") as f:
            f.write(video_file.getbuffer())
        
        # Update status
        status_placeholder.text("Loading video...")
        progress_placeholder.progress(0.1)
        
        # Load video
        video = VideoFileClip(input_path)
        
        # Extract audio
        status_placeholder.text("Extracting audio...")
        progress_placeholder.progress(0.2)
        
        audio_path = os.path.join(temp_dir, "audio.wav")
        video.audio.write_audiofile(audio_path, verbose=False)
        
        # Load Whisper model
        status_placeholder.text("Loading speech recognition model...")
        progress_placeholder.progress(0.3)
        model = whisper.load_model("base")
        
        # Transcribe
        status_placeholder.text("Transcribing audio...")
        progress_placeholder.progress(0.4)
        result = model.transcribe(audio_path)
        transcript = result["text"]
        
        # Initialize translator
        translator = Translator()
        
        # Translate to Hindi
        status_placeholder.text("Translating to Hindi...")
        progress_placeholder.progress(0.5)
        hindi_text = translator.translate(transcript, dest='hi').text
        
        # Translate to English
        status_placeholder.text("Translating to English...")
        progress_placeholder.progress(0.6)
        english_text = translator.translate(transcript, dest='en').text
        
        # Create Hindi audio
        status_placeholder.text("Creating Hindi audio...")
        progress_placeholder.progress(0.7)
        hindi_audio_path = os.path.join(temp_dir, "hindi_audio.mp3")
        hindi_tts = gTTS(text=hindi_text, lang='hi')
        hindi_tts.save(hindi_audio_path)
        
        # Create English audio
        status_placeholder.text("Creating English audio...")
        progress_placeholder.progress(0.8)
        english_audio_path = os.path.join(temp_dir, "english_audio.mp3")
        english_tts = gTTS(text=english_text, lang='en')
        english_tts.save(english_audio_path)
        
        # Create videos with new audio
        status_placeholder.text("Creating final videos...")
        progress_placeholder.progress(0.9)
        
        # Hindi version
        hindi_output_path = os.path.join(temp_dir, "output_hindi.mp4")
        hindi_video = video.set_audio(AudioFileClip(hindi_audio_path))
        hindi_video.write_videofile(hindi_output_path, codec='libx264', audio_codec='aac', verbose=False)
        
        # English version
        english_output_path = os.path.join(temp_dir, "output_english.mp4")
        english_video = video.set_audio(AudioFileClip(english_audio_path))
        english_video.write_videofile(english_output_path, codec='libx264', audio_codec='aac', verbose=False)
        
        # Read the output files
        with open(hindi_output_path, 'rb') as f:
            hindi_video_data = f.read()
        with open(english_output_path, 'rb') as f:
            english_video_data = f.read()
            
        # Clean up
        video.close()
        
        progress_placeholder.progress(1.0)
        status_placeholder.text("Processing complete!")
        
        return hindi_video_data, english_video_data
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        status_placeholder.error(f"Error: {str(e)}")
        return None, None
        
    finally:
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    st.set_page_config(page_title="Video Translator", page_icon="🎥", layout="wide")
    
    st.title("🎥 Video Translator")
    st.write("Upload a video to translate it to Hindi and English")
    
    # File uploader
    video_file = st.file_uploader("Choose a video file", type=['mp4'])
    
    if video_file is not None:
        # Create placeholders for progress and status
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Add a process button
        if st.button("Process Video"):
            try:
                # Process the video
                hindi_video, english_video = process_video(
                    video_file,
                    progress_placeholder,
                    status_placeholder
                )
                
                if hindi_video and english_video:
                    # Create download buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="Download Hindi Version",
                            data=hindi_video,
                            file_name=f"{video_file.name.split('.')[0]}_hindi.mp4",
                            mime="video/mp4"
                        )
                    
                    with col2:
                        st.download_button(
                            label="Download English Version",
                            data=english_video,
                            file_name=f"{video_file.name.split('.')[0]}_english.mp4",
                            mime="video/mp4"
                        )
                        
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                logger.error(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()
