import streamlit as st
import whisper
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment, effects
from moviepy.editor import VideoFileClip, AudioFileClip
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS
import os
import tempfile

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 AI Dubbing System",
    page_icon="🎬",
    layout="centered"
)

# ── Language config ───────────────────────────────────────────────
LANGUAGES = {
    'Hindi':   {'helsinki': 'hi', 'tts_lang': 'hi'},
    'Telugu':  {'helsinki': 'te', 'tts_lang': 'te'},
    'French':  {'helsinki': 'fr', 'tts_lang': 'fr'},
    'Spanish': {'helsinki': 'es', 'tts_lang': 'es'},
    'German':  {'helsinki': 'de', 'tts_lang': 'de'},
}

# ── Load models (cached so they load only once) ───────────────────
@st.cache_resource
def load_whisper():
    st.info("Loading Whisper model... (first time takes 1 min)")
    return whisper.load_model("base")

@st.cache_resource
def load_tts():
    st.info("Loading TTS model... (first time takes 2 mins)")
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2")

@st.cache_resource
def load_translator(lang_code):
    model_name = f"Helsinki-NLP/opus-mt-en-{lang_code}"
    tokenizer  = MarianTokenizer.from_pretrained(model_name)
    model      = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# ── Stage 1: Transcribe ───────────────────────────────────────────
def transcribe(video_path, whisper_model):
    result   = whisper_model.transcribe(video_path)
    segments = [
        {
            'start': round(s['start'], 2),
            'end':   round(s['end'],   2),
            'text':  s['text'].strip()
        }
        for s in result['segments']
    ]
    return segments

# ── Stage 2: Translate ────────────────────────────────────────────
def translate(segments, lang_code):
    tokenizer, model = load_translator(lang_code)
    translated = []
    for seg in segments:
        inputs = tokenizer(seg['text'], return_tensors="pt", padding=True)
        out    = model.generate(**inputs)
        text   = tokenizer.decode(out[0], skip_special_tokens=True)
        translated.append({
            'start': seg['start'],
            'end':   seg['end'],
            'text':  text
        })
    return translated

# ── Stage 3: TTS + timing ─────────────────────────────────────────
def synthesize(segments, tts_lang, tts_model, output_dir):
    sample_rate = 22050
    total_dur   = segments[-1]['end']
    final_audio = np.zeros(int(total_dur * sample_rate))

    for i, seg in enumerate(segments):
        wav_path = os.path.join(output_dir, f"seg_{i}.wav")
        try:
            # Generate speech
            tts_model.tts_to_file(
                text=seg['text'],
                language=tts_lang,
                file_path=wav_path
            )

            # Boost volume
            audio_seg = AudioSegment.from_wav(wav_path)
            audio_seg = audio_seg + 10
            audio_seg = effects.normalize(audio_seg)
            audio_seg.export(wav_path, format='wav')

            # Load as numpy
            audio, sr = librosa.load(wav_path, sr=sample_rate)

            # Time stretch if needed
            target_dur  = seg['end'] - seg['start']
            target_samp = int(target_dur * sample_rate)
            if target_samp <= 0:
                continue

            ratio = (len(audio) / sample_rate) / target_dur
            if ratio > 1.25 or ratio < 0.75:
                audio = librosa.effects.time_stretch(audio, rate=ratio)

            # Fit to slot
            if len(audio) > target_samp:
                audio = audio[:target_samp]
            else:
                audio = np.pad(audio, (0, target_samp - len(audio)))

            # Place at exact timestamp
            start_s = int(seg['start'] * sample_rate)
            end_s   = start_s + len(audio)
            if end_s > len(final_audio):
                audio = audio[:len(final_audio) - start_s]
                end_s = len(final_audio)
            final_audio[start_s:end_s] = audio

        except Exception as e:
            st.warning(f"Segment {i+1} skipped: {e}")
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    # Normalize final audio
    max_val = np.max(np.abs(final_audio))
    if max_val > 0:
        final_audio = final_audio / max_val * 0.95
    np.clip(final_audio, -1.0, 1.0, out=final_audio)

    out_path = os.path.join(output_dir, "dubbed_audio.wav")
    sf.write(out_path, final_audio, sample_rate)
    return out_path

# ── Stage 4: Merge video ──────────────────────────────────────────
def merge(video_path, audio_path, output_dir):
    out_path = os.path.join(output_dir, "dubbed_video.mp4")
    video    = VideoFileClip(video_path)
    audio    = AudioFileClip(audio_path)
    if audio.duration > video.duration:
        audio = audio.subclip(0, video.duration)
    final = video.set_audio(audio)
    final.write_videofile(out_path, codec='libx264',
                          audio_codec='aac', verbose=False, logger=None)
    video.close()
    audio.close()
    return out_path

# ── UI ────────────────────────────────────────────────────────────
st.title("🎬 AI Dubbing System")
st.markdown("""
Automatically dubs any **English video** into another language using AI.

**Pipeline:** Whisper → Helsinki-NLP → Coqui XTTS v2 → MoviePy
""")

st.divider()

video_file = st.file_uploader(
    "📁 Upload English Video (MP4)",
    type=["mp4", "mkv", "avi"]
)

target_language = st.selectbox(
    "🌐 Select Target Language",
    list(LANGUAGES.keys())
)

st.divider()

if st.button("🎙️ Start Dubbing", type="primary", use_container_width=True):
    if video_file is None:
        st.error("Please upload a video first!")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save uploaded video to temp file
            video_path = os.path.join(tmpdir, "input_video.mp4")
            with open(video_path, "wb") as f:
                f.write(video_file.read())

            lang_code = LANGUAGES[target_language]['helsinki']
            tts_lang  = LANGUAGES[target_language]['tts_lang']

            # Load models
            with st.spinner("Loading AI models..."):
                whisper_model = load_whisper()
                tts_model     = load_tts()

            # Stage 1
            with st.spinner("🎙️ Stage 1: Transcribing audio..."):
                segments = transcribe(video_path, whisper_model)
                st.success(f"✅ Transcribed {len(segments)} segments")

            # Stage 2
            with st.spinner(f"🌐 Stage 2: Translating to {target_language}..."):
                translated = translate(segments, lang_code)
                st.success(f"✅ Translated {len(translated)} segments")

                # Show translation preview
                with st.expander("👀 See translations"):
                    for s in translated[:5]:
                        st.write(f"**{s['start']}s:** {s['text']}")

            # Stage 3
            with st.spinner("🔊 Stage 3: Generating dubbed voice..."):
                audio_path = synthesize(translated, tts_lang, tts_model, tmpdir)
                st.success("✅ Dubbed audio generated")

            # Stage 4
            with st.spinner("🎬 Stage 4: Merging audio into video..."):
                output_path = merge(video_path, audio_path, tmpdir)
                st.success("✅ Video dubbed successfully!")

            # Show result
            st.divider()
            st.subheader("🎉 Your Dubbed Video")
            with open(output_path, "rb") as f:
                video_bytes = f.read()

            st.video(video_bytes)
            st.download_button(
                label="⬇️ Download Dubbed Video",
                data=video_bytes,
                file_name=f"dubbed_{target_language.lower()}.mp4",
                mime="video/mp4",
                use_container_width=True
            )

st.divider()
st.markdown("""
| Stage | Technology | What happens |
|-------|-----------|-------------|
| 1 | OpenAI Whisper | Speech → timestamped text |
| 2 | Helsinki-NLP | English → target language |
| 3 | Coqui XTTS v2 | Text → natural dubbed voice |
| 4 | MoviePy | Replace audio in video |

Built by **Darisa Adithya** · [GitHub](https://github.com/darisaadithya7)
""")
