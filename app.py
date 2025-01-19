import streamlit as st
import os
import whisper
import torch
from googletrans import Translator
from moviepy.editor import VideoFileClip, AudioFileClip
from gtts import gTTS
import tempfile
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
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

def process_video(video_file, processor, progress_bar, status_text):
    """Process a single video file"""
    temp_files = []  # Track temporary files for cleanup

    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded video
            temp_video_path = os.path.join(temp_dir, "input_video.mp4")
            with open(temp_video_path, "wb") as f:
                f.write(video_file.read())
            temp_files.append(temp_video_path)

            # Load video and extract audio
            status_text.text("Loading video...")
            progress_bar.progress(0.1)
            
            video = VideoFileClip(temp_video_path)
            if not video.audio:
                raise ValueError("Video has no audio track")

            # Extract audio
            status_text.text("Extracting audio...")
            progress_bar.progress(0.2)
            temp_audio_path = os.path.join(temp_dir, "audio.wav")
            video.audio.write_audiofile(temp_audio_path, 
                                      fps=16000, 
                                      verbose=False, 
                                      logger=None)
            temp_files.append(temp_audio_path)

            # Transcribe audio
            status_text.text("Transcribing audio...")
            progress_bar.progress(0.4)
            transcript = processor.transcribe_audio(temp_audio_path)

            # Translate text
            status_text.text("Translating text...")
            progress_bar.progress(0.6)
            hindi_text = processor.translate_text(transcript, 'hi')
            english_text = processor.translate_text(transcript, 'en')

            translated_videos = {}
            
            # Process Hindi version
            status_text.text("Creating Hindi version...")
            progress_bar.progress(0.8)
            hindi_audio_path = os.path.join(temp_dir, "hindi_audio.mp3")
            processor.create_audio(hindi_text, 'hi', hindi_audio_path)
            
            hindi_video = video.set_audio(AudioFileClip(hindi_audio_path))
            hindi_output_path = os.path.join(temp_dir, "output_hindi.mp4")
            hindi_video.write_videofile(hindi_output_path, 
                                      codec='libx264', 
                                      audio_codec='aac',
                                      verbose=False,
                                      logger=None)
            
            with open(hindi_output_path, 'rb') as f:
                translated_videos['hindi'] = f.read()

            # Process English version
            status_text.text("Creating English version...")
            progress_bar.progress(0.9)
            english_audio_path = os.path.join(temp_dir, "english_audio.mp3")
            processor.create_audio(english_text, 'en', english_audio_path)
            
            english_video = video.set_audio(AudioFileClip(english_audio_path))
            english_output_path = os.path.join(temp_dir, "output_english.mp4")
            english_video.write_videofile(english_output_path, 
                                        codec='libx264', 
                                        audio_codec='aac',
                                        verbose=False,
                                        logger=None)
            
            with open(english_output_path, 'rb') as f:
                translated_videos['english'] = f.read()

            # Clean up
            video.close()

            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
            
            return translated_videos

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise
    finally:
        # Cleanup temporary files
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Error deleting temporary file {file_path}: {str(e)}")

def main():
    st.set_page_config(
        page_title="Video Translator",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .status-text {
            font-size: 1.1rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎥 Video Translator")
    st.markdown("### Translate your videos to Hindi and English")

    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ Information")
        st.info(f"Running on: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
        st.markdown("""
        ### Supported Formats
        - MP4 videos
        - Maximum file size: 200MB
        
        ### Processing Steps
        1. Audio extraction
        2. Speech recognition
        3. Translation
        4. Text-to-speech
        5. Video generation
        """)

    try:
        # Initialize processor
        processor = VideoProcessor()

        # File upload
        uploaded_files = st.file_uploader(
            "Upload your videos (MP4 format)",
            type=["mp4"],
            accept_multiple_files=True,
            help="Maximum file size: 200MB"
        )

        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} file(s) uploaded")

            for uploaded_file in uploaded_files:
                # Check file size
                if uploaded_file.size > 200 * 1024 * 1024:  # 200MB limit
                    st.error(f"❌ {uploaded_file.name} exceeds 200MB limit")
                    continue

                # Process each video
                with st.expander(f"📽️ Processing: {uploaded_file.name}", expanded=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        translated_videos = process_video(
                            uploaded_file,
                            processor,
                            progress_bar,
                            status_text
                        )

                        if translated_videos:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.download_button(
                                    "⬇️ Download Hindi Version",
                                    data=translated_videos['hindi'],
                                    file_name=f"{uploaded_file.name.split('.')[0]}_hindi.mp4",
                                    mime="video/mp4",
                                    key=f"hindi_{uploaded_file.name}"
                                )
                            
                            with col2:
                                st.download_button(
                                    "⬇️ Download English Version",
                                    data=translated_videos['english'],
                                    file_name=f"{uploaded_file.name.split('.')[0]}_english.mp4",
                                    mime="video/mp4",
                                    key=f"english_{uploaded_file.name}"
                                )
                            
                            st.success("✅ Processing completed successfully!")

                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                        logger.error(f"Error processing {uploaded_file.name}: {str(e)}")

        # Instructions
        if not uploaded_files:
            st.markdown("""
            ### 🎯 How to use:
            1. Upload one or more MP4 videos using the file uploader above
            2. Wait for the processing to complete
            3. Download the translated versions
            
            ### ⚠️ Note:
            - Processing time depends on video length and system resources
            - Make sure your videos have clear audio for best results
            - Internet connection is required for translation
            """)

    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        logger.error(f"Application Error: {str(e)}")

if __name__ == "__main__":
    main()
