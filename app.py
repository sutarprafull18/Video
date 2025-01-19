import streamlit as st
import os
import tempfile
import logging
from pathlib import Path
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Check and install ffmpeg if needed
def setup_ffmpeg():
    try:
        # Try to install ffmpeg using apt-get
        subprocess.run(['apt-get', 'update'], check=True)
        subprocess.run(['apt-get', 'install', '-y', 'ffmpeg'], check=True)
    except Exception as e:
        st.error(f"Error installing ffmpeg: {str(e)}")
        logger.error(f"Error installing ffmpeg: {str(e)}")

# Initialize application
def init_app():
    try:
        setup_ffmpeg()
        
        # Import required packages
        global VideoFileClip, AudioFileClip, whisper, Translator, gTTS
        from moviepy.editor import VideoFileClip, AudioFileClip
        import whisper
        from googletrans import Translator
        from gtts import gTTS
        
        return True
    except Exception as e:
        st.error(f"Error initializing application: {str(e)}")
        logger.error(f"Error initializing application: {str(e)}")
        return False

# Now import the required packages

# Configure logging


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

def process_video(video_file, progress_bar, status_text):
    """Process a single video file"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded video
        input_path = os.path.join(temp_dir, "input_video.mp4")
        with open(input_path, "wb") as f:
            f.write(video_file.getbuffer())
        
        status_text.text("Loading video...")
        progress_bar.progress(0.1)
        
        # Load video and check audio
        video = VideoFileClip(input_path)
        if not video.audio:
            raise ValueError("No audio track found in video")
        
        # Extract audio
        status_text.text("Extracting audio...")
        progress_bar.progress(0.2)
        audio_path = os.path.join(temp_dir, "audio.wav")
        video.audio.write_audiofile(audio_path, fps=16000, verbose=False)
        
        # Load model and transcribe
        status_text.text("Transcribing audio...")
        progress_bar.progress(0.4)
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        transcript = result["text"]
        
        # Translate
        status_text.text("Translating...")
        progress_bar.progress(0.6)
        translator = Translator()
        hindi_text = translator.translate(transcript, dest='hi').text
        english_text = translator.translate(transcript, dest='en').text
        
        # Generate audio for translations
        translated_videos = {}
        
        # Hindi version
        status_text.text("Creating Hindi version...")
        progress_bar.progress(0.8)
        hindi_audio_path = os.path.join(temp_dir, "hindi.mp3")
        hindi_tts = gTTS(text=hindi_text, lang='hi')
        hindi_tts.save(hindi_audio_path)
        
        hindi_video = video.set_audio(AudioFileClip(hindi_audio_path))
        hindi_output = os.path.join(temp_dir, "output_hindi.mp4")
        hindi_video.write_videofile(hindi_output, 
                                  codec='libx264', 
                                  audio_codec='aac',
                                  verbose=False)
        
        # English version
        status_text.text("Creating English version...")
        progress_bar.progress(0.9)
        english_audio_path = os.path.join(temp_dir, "english.mp3")
        english_tts = gTTS(text=english_text, lang='en')
        english_tts.save(english_audio_path)
        
        english_video = video.set_audio(AudioFileClip(english_audio_path))
        english_output = os.path.join(temp_dir, "output_english.mp4")
        english_video.write_videofile(english_output, 
                                    codec='libx264', 
                                    audio_codec='aac',
                                    verbose=False)
        
        # Read output files
        with open(hindi_output, 'rb') as f:
            hindi_data = f.read()
        with open(english_output, 'rb') as f:
            english_data = f.read()
        
        # Cleanup
        video.close()
        
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        return hindi_data, english_data
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        status_text.error(f"Error: {str(e)}")
        return None, None
        
    finally:
        # Cleanup temp directory
        import shutil
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            logger.error(f"Error cleaning up: {str(e)}")

def main():
    st.set_page_config(
        page_title="Video Translator",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 Video Translator")
    
    # Initialize app
    if not init_app():
        st.error("Failed to initialize application. Please check the logs.")
        return
    
    # Add file size warning
    st.warning("⚠️ Maximum file size: 200MB")
    
    # File uploader
    video_file = st.file_uploader(
        "Upload your video (MP4 format)",
        type=['mp4'],
        help="Upload an MP4 video file to translate"
    )
    
    if video_file:
        # Check file size (200MB limit)
        if video_file.size > 200 * 1024 * 1024:
            st.error("File too large! Maximum size is 200MB")
            return
            
        # Show file info
        st.info(f"File: {video_file.name} ({video_file.size / 1024 / 1024:.1f} MB)")
        
        if st.button("Start Translation"):
            with st.spinner("Processing video..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    hindi_video, english_video = process_video(
                        video_file,
                        progress_bar,
                        status_text
                    )
                    
                    if hindi_video and english_video:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.download_button(
                                "⬇️ Download Hindi Version",
                                data=hindi_video,
                                file_name=f"{video_file.name.split('.')[0]}_hindi.mp4",
                                mime="video/mp4"
                            )
                        
                        with col2:
                            st.download_button(
                                "⬇️ Download English Version",
                                data=english_video,
                                file_name=f"{video_file.name.split('.')[0]}_english.mp4",
                                mime="video/mp4"
                            )
                            
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    logger.error(f"Error processing video: {str(e)}")
    
    # Instructions
    with st.expander("ℹ️ Instructions & Tips"):
        st.markdown("""
        ### How to use:
        1. Upload an MP4 video file (max 200MB)
        2. Click 'Start Translation'
        3. Wait for processing to complete
        4. Download the translated versions
        
        ### Tips:
        - Ensure your video has clear audio
        - Shorter videos process faster
        - Keep the video under 200MB
        - MP4 format is required
        """)

if __name__ == "__main__":
    main()

