import streamlit as st
import whisper
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment, effects
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
from transformers import MarianMTModel, MarianTokenizer
from googletrans import Translator
from gtts import gTTS
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
    'Hindi':   {'helsinki': 'hi', 'gtts_lang': 'hi', 'google_code': 'hi', 'use_google': False},
    'Telugu':  {'helsinki': None, 'gtts_lang': 'te', 'google_code': 'te', 'use_google': True},
    'French':  {'helsinki': 'fr', 'gtts_lang': 'fr', 'google_code': 'fr', 'use_google': False},
    'Spanish': {'helsinki': 'es', 'gtts_lang': 'es', 'google_code': 'es', 'use_google': False},
    'German':  {'helsinki': 'de', 'gtts_lang': 'de', 'google_code': 'de', 'use_google': False},
}

# ── Load models ───────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_whisper():
    return whisper.load_model("tiny")

@st.cache_resource(show_spinner=False)
def load_helsinki(lang_code):
    model_name = f"Helsinki-NLP/opus-mt-en-{lang_code}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# ── Stage 1: Transcribe ───────────────────────────────────────────

def transcribe(video_path, whisper_model):
    result = whisper_model.transcribe(video_path, fp16=False)
    segments = [
        {
            'start': round(s['start'], 2),
            'end':   round(s['end'],   2),
            'text':  s['text'].strip()
        }
        for s in result['segments']
        if s['text'].strip()
    ]
    return segments

# ── Stage 2: Translate ────────────────────────────────────────────

def translate_google(segments, google_code):
    translator = Translator()
    translated = []
    for seg in segments:
        try:
            result = translator.translate(seg['text'], dest=google_code)
            translated.append({'start': seg['start'], 'end': seg['end'], 'text': result.text})
        except Exception as e:
            st.warning(f"Google translation skipped: {seg['text'][:30]}... ({e})")
    return translated

def translate_helsinki(segments, lang_code):
    tokenizer, model = load_helsinki(lang_code)
    translated = []
    for seg in segments:
        try:
            inputs = tokenizer(seg['text'], return_tensors="pt", padding=True, truncation=True, max_length=512)
            out = model.generate(**inputs)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            translated.append({'start': seg['start'], 'end': seg['end'], 'text': text})
        except Exception as e:
            st.warning(f"Translation skipped: {seg['text'][:30]}... ({e})")
    return translated

def translate(segments, lang_config):
    if lang_config['use_google']:
        return translate_google(segments, lang_config['google_code'])
    else:
        return translate_helsinki(segments, lang_config['helsinki'])

# ── Stage 3: TTS + timing ─────────────────────────────────────────

def synthesize(segments, gtts_lang, output_dir):
    sample_rate = 22050
    if not segments:
        raise ValueError("No segments to synthesize.")

    total_dur = segments[-1]['end']
    final_audio = np.zeros(int(total_dur * sample_rate))

    for i, seg in enumerate(segments):
        mp3_path = os.path.join(output_dir, f"seg_{i}.mp3")
        wav_path = os.path.join(output_dir, f"seg_{i}.wav")
        try:
            tts = gTTS(text=seg['text'], lang=gtts_lang, slow=False)
            tts.save(mp3_path)

            audio_seg = AudioSegment.from_mp3(mp3_path)
            audio_seg = audio_seg + 15
            audio_seg = effects.normalize(audio_seg)
            audio_seg.export(wav_path, format='wav')

            audio, sr = librosa.load(wav_path, sr=sample_rate)

            target_dur  = seg['end'] - seg['start']
            target_samp = int(target_dur * sample_rate)
            if target_samp <= 0 or len(audio) == 0:
                continue

            ratio = (len(audio) / sample_rate) / target_dur
            if ratio > 1.25 or ratio < 0.75:
                audio = librosa.effects.time_stretch(audio, rate=float(ratio))

            if len(audio) > target_samp:
                audio = audio[:target_samp]
            else:
                audio = np.pad(audio, (0, target_samp - len(audio)))

            start_s = int(seg['start'] * sample_rate)
            end_s   = start_s + len(audio)
            if start_s >= len(final_audio):
                continue
            if end_s > len(final_audio):
                audio = audio[:len(final_audio) - start_s]
                end_s = len(final_audio)

            final_audio[start_s:end_s] = audio

        except Exception as e:
            st.warning(f"Segment {i+1} skipped: {e}")
        finally:
            for f in [mp3_path, wav_path]:
                if os.path.exists(f):
                    os.remove(f)

    max_val = np.max(np.abs(final_audio))
    if max_val > 0:
        final_audio = final_audio / max_val * 0.95
    np.clip(final_audio, -1.0, 1.0, out=final_audio)

    out_path = os.path.join(output_dir, "dubbed_audio.wav")
    sf.write(out_path, final_audio, sample_rate)
    return out_path

# ── Stage 4a: Generate SRT subtitle file ─────────────────────────

def format_srt_time(seconds):
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm"""
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def generate_srt(segments, output_dir):
    """Generate a .srt subtitle file from translated segments."""
    srt_path = os.path.join(output_dir, "subtitles.srt")
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = format_srt_time(seg['start'])
        end   = format_srt_time(seg['end'])
        lines.append(f"{i}\n{start} --> {end}\n{seg['text']}\n")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return srt_path

# ── Stage 4b: Burn captions into video ───────────────────────────

def burn_captions(video_path, segments, output_dir):
    """Burn translated subtitles as captions directly onto the video frames."""
    video = VideoFileClip(video_path)
    caption_clips = []

    for seg in segments:
        try:
            duration = seg['end'] - seg['start']
            if duration <= 0:
                continue

            txt_clip = (
                TextClip(
                    seg['text'],
                    fontsize=28,
                    font='DejaVu-Sans',        # font available on Linux/HF Spaces
                    color='white',
                    stroke_color='black',
                    stroke_width=1.5,
                    method='caption',
                    size=(int(video.w * 0.9), None),  # 90% of video width, auto height
                    align='center'
                )
                .set_start(seg['start'])
                .set_duration(duration)
                .set_position(('center', 0.85), relative=True)  # bottom 15% of screen
            )
            caption_clips.append(txt_clip)
        except Exception as e:
            st.warning(f"Caption skipped for segment: {seg['text'][:30]}... ({e})")

    if not caption_clips:
        return None

    out_path = os.path.join(output_dir, "captioned_video.mp4")
    final = CompositeVideoClip([video] + caption_clips)
    final.write_videofile(
        out_path,
        codec='libx264',
        audio_codec='aac',
        verbose=False,
        logger=None
    )
    video.close()
    return out_path

# ── Stage 5: Merge dubbed audio into video ────────────────────────

def merge(video_path, audio_path, output_dir):
    out_path = os.path.join(output_dir, "dubbed_video.mp4")
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    if audio.duration > video.duration:
        audio = audio.subclip(0, video.duration)

    final = video.set_audio(audio)
    final.write_videofile(
        out_path,
        codec='libx264',
        audio_codec='aac',
        verbose=False,
        logger=None
    )
    video.close()
    audio.close()
    return out_path

# ── UI ────────────────────────────────────────────────────────────

st.title("🎬 AI Dubbing System")
st.markdown("""
Automatically dubs any **English video** into another language using AI.

**Pipeline:** Whisper → Translation → gTTS → Captions → MoviePy

> ⚠️ **Keep videos under 2 minutes** for best performance.
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

lang_config = LANGUAGES[target_language]
if lang_config['use_google']:
    st.info("ℹ️ Telugu uses Google Translate for accurate translation.")

# Caption options
st.markdown("#### 📝 Caption Options")
col1, col2 = st.columns(2)
with col1:
    burn_captions_on = st.checkbox("🔥 Burn captions into video", value=True,
                                    help="Captions will be permanently visible on the video")
with col2:
    download_srt = st.checkbox("📄 Download .srt subtitle file", value=True,
                                help="Standard subtitle file, works with VLC, YouTube, etc.")

st.divider()

if st.button("🎙️ Start Dubbing", type="primary", use_container_width=True):
    if video_file is None:
        st.error("Please upload a video first!")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:

            video_path = os.path.join(tmpdir, "input_video.mp4")
            with open(video_path, "wb") as f:
                f.write(video_file.read())

            gtts_lang = lang_config['gtts_lang']

            # Stage 1 — Transcribe
            with st.spinner("🎙️ Stage 1: Transcribing audio..."):
                try:
                    whisper_model = load_whisper()
                    segments = transcribe(video_path, whisper_model)
                    st.success(f"✅ Transcribed {len(segments)} segments")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    st.stop()

            if not segments:
                st.error("No speech detected in the video. Please try a different video.")
                st.stop()

            # Stage 2 — Translate
            with st.spinner(f"🌐 Stage 2: Translating to {target_language}..."):
                try:
                    translated = translate(segments, lang_config)
                    st.success(f"✅ Translated {len(translated)} segments")
                except Exception as e:
                    st.error(f"Translation failed: {e}")
                    st.stop()

            with st.expander("👀 See translations"):
                for s in translated[:5]:
                    st.write(f"**{s['start']}s – {s['end']}s:** {s['text']}")

            # Stage 3 — TTS
            with st.spinner("🔊 Stage 3: Generating dubbed audio..."):
                try:
                    audio_path = synthesize(translated, gtts_lang, tmpdir)
                    st.success("✅ Dubbed audio generated")
                except Exception as e:
                    st.error(f"Audio synthesis failed: {e}")
                    st.stop()

            # Stage 4 — Merge audio
            with st.spinner("🎬 Stage 4: Merging dubbed audio into video..."):
                try:
                    dubbed_path = merge(video_path, audio_path, tmpdir)
                    st.success("✅ Audio merged")
                except Exception as e:
                    st.error(f"Video merge failed: {e}")
                    st.stop()

            # Stage 5 — Captions
            srt_bytes    = None
            captioned_path = dubbed_path  # default: no burned captions

            if download_srt:
                with st.spinner("📄 Stage 5a: Generating .srt subtitle file..."):
                    try:
                        srt_path = generate_srt(translated, tmpdir)
                        with open(srt_path, "rb") as f:
                            srt_bytes = f.read()
                        st.success("✅ SRT file ready")
                    except Exception as e:
                        st.warning(f"SRT generation failed: {e}")

            if burn_captions_on:
                with st.spinner("🔥 Stage 5b: Burning captions into video..."):
                    try:
                        result = burn_captions(dubbed_path, translated, tmpdir)
                        if result:
                            captioned_path = result
                            st.success("✅ Captions burned into video")
                        else:
                            st.warning("No captions were added (all segments skipped).")
                    except Exception as e:
                        st.warning(f"Caption burning failed (video still available without captions): {e}")

            # ── Results ──────────────────────────────────────────
            st.divider()
            st.subheader("🎉 Your Dubbed Video")

            with open(captioned_path, "rb") as f:
                video_bytes = f.read()

            st.video(video_bytes)

            # Download buttons side by side
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    label="⬇️ Download Dubbed Video",
                    data=video_bytes,
                    file_name=f"dubbed_{target_language.lower()}.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
            with dl_col2:
                if srt_bytes:
                    st.download_button(
                        label="📄 Download .srt Subtitles",
                        data=srt_bytes,
                        file_name=f"subtitles_{target_language.lower()}.srt",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st.info("Enable SRT option above to get subtitle file.")

st.divider()
st.markdown("""
| Stage | Technology | What happens |
|-------|-----------|-------------|
| 1 | OpenAI Whisper (tiny) | Speech → timestamped text |
| 2 | Helsinki-NLP / Google Translate | English → target language |
| 3 | gTTS | Text → dubbed audio |
| 4 | MoviePy | Replace audio in video |
| 5 | MoviePy TextClip + SRT | Burn captions & generate subtitle file |

Built by **Darisa Adithya** · [GitHub](https://github.com/darisaadithya7)
""")
