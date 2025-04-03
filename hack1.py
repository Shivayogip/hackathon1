import sys
import os
import queue
import json
import vosk
import sounddevice as sd
from deep_translator import GoogleTranslator
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QTextEdit, QVBoxLayout, QLabel, QComboBox, QHBoxLayout
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QThread, pyqtSignal
from gtts import gTTS
import pygame
from fpdf import FPDF

# Initialize Vosk Model
MODEL_PATH = "model"
if not os.path.exists(MODEL_PATH):
    print("Please download the Vosk model and place it in the 'model' folder.")
    sys.exit(1)
model = vosk.Model(MODEL_PATH)

audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(bytes(indata))

# Thread for processing audio
class AudioProcessingThread(QThread):
    transcription_signal = pyqtSignal(str)
    summary_signal = pyqtSignal(str)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.recording = True

    def run(self):
        recognizer = vosk.KaldiRecognizer(self.model, 16000)
        transcription = ""
        while self.recording:
            if not audio_queue.empty():
                data = audio_queue.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "")
                    transcription += text + " "
                    self.transcription_signal.emit(transcription)

        summary = self.generate_summary(transcription)
        self.summary_signal.emit(summary)

    def generate_summary(self, text):
        if len(text.split()) < 10:
            return "Summary: Not enough data for summarization."

        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary_sentences = summarizer(parser.document, 2)
        summary = " ".join(str(sentence) for sentence in summary_sentences)
        return "Summary: " + summary

    def stop(self):
        self.recording = False

# Main App
class MeetingSummarizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("🎤 AI Meeting Summarizer & Translator 🌍")
        self.setGeometry(200, 100, 700, 600)

        self.title_label = QLabel("📢 AI Meeting Summarizer")
        self.title_label.setFont(QFont("Arial", 16, QFont.Bold))

        self.transcription_output = QTextEdit(self)
        self.transcription_output.setReadOnly(True)

        self.summary_output = QTextEdit(self)
        self.summary_output.setReadOnly(True)

        self.language_selector = QComboBox(self)
        self.language_selector.addItems(["hi (Hindi)", "kn (Kannada)", "es (Spanish)", "fr (French)", "de (German)"])
        
        self.translate_btn = QPushButton("🌍 Translate Summary", self)
        self.translate_btn.clicked.connect(self.translate_summary)
        
        self.translation_output = QTextEdit(self)
        self.translation_output.setReadOnly(True)

        self.read_summary_btn = QPushButton("🔊 Read Summary", self)
        self.read_summary_btn.clicked.connect(self.read_summary)

        self.read_translation_btn = QPushButton("🔊 Read Translation", self)
        self.read_translation_btn.clicked.connect(self.read_translation)

        self.download_pdf_btn = QPushButton("📄 Download Summary as PDF", self)
        self.download_pdf_btn.clicked.connect(self.download_summary_pdf)

        self.record_btn = QPushButton("🎙 Start Recording", self)
        self.record_btn.clicked.connect(self.start_recording)

        self.stop_btn = QPushButton("🛑 Stop Recording", self)
        self.stop_btn.clicked.connect(self.stop_recording)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.record_btn)
        button_layout.addWidget(self.stop_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addWidget(QLabel("📝 Transcription:"))
        layout.addWidget(self.transcription_output)
        layout.addWidget(QLabel("📄 Summary:"))
        layout.addWidget(self.summary_output)
        layout.addWidget(QLabel("🌍 Select Language:"))
        layout.addWidget(self.language_selector)
        layout.addWidget(self.translate_btn)
        layout.addWidget(QLabel("🔄 Translated Summary:"))
        layout.addWidget(self.translation_output)
        layout.addWidget(self.read_summary_btn)
        layout.addWidget(self.read_translation_btn)
        layout.addWidget(self.download_pdf_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        pygame.init()

    def start_recording(self):
        self.transcription_output.setText("🎤 Recording... Speak Now!")
        self.stream = sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback)
        self.stream.start()
        self.audio_thread = AudioProcessingThread(model)
        self.audio_thread.transcription_signal.connect(self.update_transcription)
        self.audio_thread.summary_signal.connect(self.update_summary)
        self.audio_thread.start()

    def stop_recording(self):
        self.stream.stop()
        self.stream.close()
        if hasattr(self, 'audio_thread') and self.audio_thread.isRunning():
            self.audio_thread.stop()
            self.audio_thread.quit()
            self.audio_thread.wait()

    def update_transcription(self, text):
        self.transcription_output.setText(text)

    def update_summary(self, summary):
        self.summary_output.setText(summary)

    def translate_summary(self):
        summary_text = self.summary_output.toPlainText()
        target_lang = self.language_selector.currentText().split(" ")[0]
        if summary_text:
            translated_text = GoogleTranslator(source="auto", target=target_lang).translate(summary_text)
            self.translation_output.setText(translated_text)

    def read_summary(self):
        text = self.summary_output.toPlainText()
        tts = gTTS(text)
        tts.save("summary.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load("summary.mp3")
        pygame.mixer.music.play()

    def read_translation(self):
        text = self.translation_output.toPlainText()
        tts = gTTS(text)
        tts.save("translation.mp3")
        pygame.mixer.music.load("translation.mp3")
        pygame.mixer.music.play()

    def download_summary_pdf(self):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, self.summary_output.toPlainText())
        pdf.output("summary.pdf")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MeetingSummarizerApp()
    mainWin.show()
    sys.exit(app.exec_())
