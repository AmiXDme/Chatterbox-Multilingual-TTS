import random
import numpy as np
import torch
from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
from src.audiobook.voice_management import *
from src.audiobook.project_management import *
from src.audiobook.config import config
from src.audiobook.script_processor import ScriptProcessor, create_sample_script
import gradio as gr
import spaces
import os
import json
import shutil

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

# --- Global Model Initialization ---
MODEL = None

LANGUAGE_CONFIG = {
    "ar": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac",
        "text": "في الشهر الماضي، وصلنا إلى معلم جديد بمليارين من المشاهدات على قناتنا على يوتيوب."
    },
    "da": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/da_m1.flac",
        "text": "Sidste måned nåede vi en ny milepæl med to milliarder visninger på vores YouTube-kanal."
    },
    "de": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
        "text": "Letzten Monat haben wir einen neuen Meilenstein erreicht: zwei Milliarden Aufrufe auf unserem YouTube-Kanal."
    },
    "el": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/el_m.flac",
        "text": "Τον περασμένο μήνα, φτάσαμε σε ένα νέο ορόσημο με δύο δισεκατομμύρια προβολές στο κανάλι μας στο YouTube."
    },
    "en": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
        "text": "Last month, we reached a new milestone with two billion views on our YouTube channel."
    },
    "es": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
        "text": "El mes pasado alcanzamos un nuevo hito: dos mil millones de visualizaciones en nuestro canal de YouTube."
    },
    "fi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fi_m.flac",
        "text": "Viime kuussa saavutimme uuden virstanpylvään kahden miljardin katselukerran kanssa YouTube-kanavallamme."
    },
    "fr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
        "text": "Le mois dernier, nous avons atteint un nouveau jalon avec deux milliards de vues sur notre chaîne YouTube."
    },
    "he": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/he_m1.flac",
        "text": "בחודש שעבר הגענו לאבן דרך חדשה עם שני מיליארד צפיות בערוץ היוטיוב שלנו."
    },
    "hi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac",
        "text": "पिछले महीने हमने एक नया मील का पत्थर छुआ: हमारे YouTube चैनल पर दो अरब व्यूज़।"
    },
    "it": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac",
        "text": "Il mese scorso abbiamo raggiunto un nuovo traguardo: due miliardi di visualizzazioni sul nostro canale YouTube."
    },
    "ja": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja/ja_prompts1.flac",
        "text": "先月、私たちのYouTubeチャンネルで二十億回の再生回数という新たなマイルストーンに到達しました。"
    },
    "ko": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ko_f.flac",
        "text": "지난달 우리는 유튜브 채널에서 이십억 조회수라는 새로운 이정표에 도달했습니다."
    },
    "ms": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ms_f.flac",
        "text": "Bulan lepas, kami mencapai pencapaian baru dengan dua bilion tontonan di saluran YouTube kami."
    },
    "nl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/nl_m.flac",
        "text": "Vorige maand bereikten we een nieuwe mijlpaal met twee miljard weergaven op ons YouTube-kanaal."
    },
    "no": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/no_f1.flac",
        "text": "Forrige måned nådde vi en ny milepæl med to milliarder visninger på YouTube-kanalen vår."
    },
    "pl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pl_m.flac",
        "text": "W zeszłym miesiącu osiągnęliśmy nowy kamień milowy z dwoma miliardami wyświetleń na naszym kanale YouTube."
    },
    "pt": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac",
        "text": "No mês passado, alcançámos um novo marco: dois mil milhões de visualizações no nosso canal do YouTube."
    },
    "ru": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac",
        "text": "В прошлом месяце мы достигли нового рубежа: два миллиарда просмотров на нашем YouTube-канале."
    },
    "sv": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sv_f.flac",
        "text": "Förra månaden nådde vi en ny milstolpe med två miljarder visningar på vår YouTube-kanal."
    },
    "sw": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sw_m.flac",
        "text": "Mwezi uliopita, tulifika hatua mpya ya maoni ya bilioni mbili kweny kituo chetu cha YouTube."
    },
    "tr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/tr_m.flac",
        "text": "Geçen ay YouTube kanalımızda iki milyar görüntüleme ile yeni bir dönüm noktasına ulaştık."
    },
    "zh": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f2.flac",
        "text": "上个月，我们达到了一个新的里程碑. 我们的YouTube频道观看次数达到了二十亿次，这绝对令人难以置信。"
    },
}

# --- UI Helpers ---
def default_audio_for_ui(lang: str) -> str | None:
    return LANGUAGE_CONFIG.get(lang, {}).get("audio")

def default_text_for_ui(lang: str) -> str:
    return LANGUAGE_CONFIG.get(lang, {}).get("text", "")

def get_supported_languages_display() -> str:
    """Generate a formatted display of all supported languages."""
    language_items = []
    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        language_items.append(f"**{name}** (`{code}`)")
    
    # Split into 2 lines
    mid = len(language_items) // 2
    line1 = " • ".join(language_items[:mid])
    line2 = " • ".join(language_items[mid:])
    
    return f"""
### 🌍 Supported Languages ({len(SUPPORTED_LANGUAGES)} total)
{line1}

{line2}
"""

def get_or_load_model():
    """Loads the ChatterboxMultilingualTTS model if it hasn't been loaded already,
    and ensures it's on the correct device."""
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        try:
            MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
            if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
                MODEL.to(DEVICE)
            print(f"Model loaded successfully. Internal device: {getattr(MODEL, 'device', 'N/A')}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return MODEL

# Attempt to load the model at startup.
try:
    get_or_load_model()
except Exception as e:
    print(f"CRITICAL: Failed to load model on startup. Application may not function. Error: {e}")

def set_seed(seed: int):
    """Sets the random seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def resolve_audio_prompt(language_id: str, provided_path: str | None) -> str | None:
    """
    Decide which audio prompt to use:
    - If user provided a path (upload/mic/url), use it.
    - Else, fall back to language-specific default (if any).
    """
    if provided_path and str(provided_path).strip():
        return provided_path
    return LANGUAGE_CONFIG.get(language_id, {}).get("audio")

@spaces.GPU
def generate_tts_audio(
    text_input: str,
    language_id: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5
) -> tuple[int, np.ndarray]:
    """
    Generate high-quality speech audio from text using Chatterbox Multilingual model with optional reference audio styling.
    Supported languages: English, French, German, Spanish, Italian, Portuguese, and Hindi.
    
    This tool synthesizes natural-sounding speech from input text. When a reference audio file 
    is provided, it captures the speaker's voice characteristics and speaking style. The generated audio 
    maintains the prosody, tone, and vocal qualities of the reference speaker, or uses default voice if no reference is provided.

    Args:
        text_input (str): The text to synthesize into speech
        language_id (str): The language code for synthesis (eg. en, fr, de, es, it, pt, hi)
        audio_prompt_path_input (str, optional): File path or URL to the reference audio file that defines the target voice style. Defaults to None.
        exaggeration_input (float, optional): Controls speech expressiveness (0.25-2.0, neutral=0.5, extreme values may be unstable). Defaults to 0.5.
        temperature_input (float, optional): Controls randomness in generation (0.05-5.0, higher=more varied). Defaults to 0.8.
        seed_num_input (int, optional): Random seed for reproducible results (0 for random generation). Defaults to 0.
        cfgw_input (float, optional): CFG/Pace weight controlling generation guidance (0.2-1.0). Defaults to 0.5, 0 for language transfer. 

    Returns:
        tuple[int, np.ndarray]: A tuple containing the sample rate (int) and the generated audio waveform (numpy.ndarray)
    """
    current_model = get_or_load_model()

    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    print(f"Generating audio for text: '{text_input[:50]}...'")
    
    # Handle optional audio prompt
    chosen_prompt = audio_prompt_path_input or default_audio_for_ui(language_id)

    generate_kwargs = {
        "exaggeration": exaggeration_input,
        "temperature": temperature_input,
        "cfg_weight": cfgw_input,
    }
    if chosen_prompt:
        generate_kwargs["audio_prompt_path"] = chosen_prompt
        print(f"Using audio prompt: {chosen_prompt}")
    else:
        print("No audio prompt provided; using default voice.")
        
    wav = current_model.generate(
        text_input,  # No text limit
        language_id=language_id,
        **generate_kwargs
    )
    print("Audio generation complete.")
    return (current_model.sr, wav.squeeze(0).numpy())

# --- Audiobook Functions ---
def refresh_voice_library():
    """Refresh the voice library display."""
    profiles = get_voice_profiles(config.voice_library_path)
    choices = [p.get('display_name', p['voice_name']) for p in profiles]
    info_html = get_voice_info_html(config.voice_library_path)
    return choices, info_html

def save_voice_profile_func(voice_name, display_name, description, language_id, 
                          exaggeration, cfg_weight, temperature, reference_audio):
    """Save a new voice profile."""
    if not voice_name or not display_name:
        return "Error: Voice name and display name are required!", ""
    
    profile_data = {
        "display_name": display_name,
        "description": description,
        "language": language_id,
        "exaggeration": exaggeration,
        "cfg_weight": cfg_weight,
        "temperature": temperature,
        "created": str(pd.Timestamp.now())
    }
    
    success = save_voice_profile(config.voice_library_path, voice_name, profile_data)
    if success and reference_audio:
        copy_reference_audio(config.voice_library_path, voice_name, reference_audio)
    
    if success:
        choices, info_html = refresh_voice_library()
        return f"Voice '{display_name}' saved successfully!", info_html
    else:
        return "Error saving voice profile!", ""

def load_voice_settings(voice_display_name):
    """Load settings from a saved voice profile."""
    if not voice_display_name:
        return None, 0.5, 0.5, 0.8, "No voice selected"
    
    profiles = get_voice_profiles(config.voice_library_path)
    selected_profile = None
    
    for profile in profiles:
        if profile.get('display_name') == voice_display_name:
            selected_profile = profile
            break
    
    if selected_profile:
        ref_audio_path = os.path.join(config.voice_library_path, 
                                    selected_profile['voice_name'], "reference.wav")
        if not os.path.exists(ref_audio_path):
            ref_audio_path = os.path.join(config.voice_library_path, 
                                        selected_profile['voice_name'], "reference.mp3")
        
        return (
            ref_audio_path if os.path.exists(ref_audio_path) else None,
            selected_profile.get('exaggeration', 0.5),
            selected_profile.get('cfg_weight', 0.5),
            selected_profile.get('temperature', 0.8),
            selected_profile.get('description', '')
        )
    
    return None, 0.5, 0.5, 0.8, "Voice not found"

# Create tabs interface
with gr.Blocks() as demo:
    gr.Markdown("""
    # 🎧 Chatterbox Multilingual TTS - Audiobook Edition
    Generate high-quality multilingual speech from text with reference audio styling and voice library management.
    
    For a hosted version of Chatterbox Multilingual and for finetuning, please visit [resemble.ai](https://app.resemble.ai)
    """)
    
    with gr.Tabs():
        # --- TTS Tab ---
        with gr.TabItem("🎙️ TTS Generation"):
            gr.Markdown(get_supported_languages_display())
            with gr.Row():
                with gr.Column():
                    initial_lang = "fr"
                    text = gr.Textbox(
                        value=default_text_for_ui(initial_lang),
                        label="Text to synthesize",
                        max_lines=10,
                        lines=5
                    )
                    
                    language_id = gr.Dropdown(
                        choices=list(ChatterboxMultilingualTTS.get_supported_languages().keys()),
                        value=initial_lang,
                        label="Language",
                        info="Select the language for text-to-speech synthesis"
                    )
                    
                    saved_voices = gr.Dropdown(
                        choices=refresh_voice_library()[0],
                        label="Saved Voices",
                        info="Load settings from saved voice profiles",
                        value=None
                    )
                    
                    ref_wav = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Reference Audio File (Optional)",
                        value=default_audio_for_ui(initial_lang)
                    )
                    
                    gr.Markdown(
                        "💡 **Note**: Ensure that the reference clip matches the specified language tag. Otherwise, language transfer outputs may inherit the accent of the reference clip's language. To mitigate this, set the CFG weight to 0.",
                        elem_classes=["audio-note"]
                    )
                    
                    exaggeration = gr.Slider(
                        0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5
                    )
                    cfg_weight = gr.Slider(
                        0.2, 1, step=.05, label="CFG/Pace", value=0.5
                    )

                    with gr.Accordion("More options", open=False):
                        seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                        temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=.8)

                    run_btn = gr.Button("Generate", variant="primary")

                with gr.Column():
                    audio_output = gr.Audio(label="Output Audio")

                def on_language_change(lang, current_ref, current_text):
                    return default_audio_for_ui(lang), default_text_for_ui(lang)

                def on_voice_change(voice_name):
                    if voice_name:
                        ref, exag, cfg, temp, desc = load_voice_settings(voice_name)
                        return ref, exag, cfg, temp, desc
                    return None, 0.5, 0.5, 0.8, ""

                language_id.change(
                    fn=on_language_change,
                    inputs=[language_id, ref_wav, text],
                    outputs=[ref_wav, text],
                    show_progress=False
                )
                
                saved_voices.change(
                    fn=on_voice_change,
                    inputs=[saved_voices],
                    outputs=[ref_wav, exaggeration, cfg_weight, temp]
                )

        # --- Voice Library Tab ---
        with gr.TabItem("📚 Voice Library"):
            gr.Markdown("""
            ## 🎭 Voice Library Management
            Create and manage voice profiles for consistent character voices across your audiobook projects.
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### ➕ Create New Voice Profile")
                    
                    voice_name = gr.Textbox(
                        label="Voice Name (folder name)",
                        placeholder="e.g., narrator_deep, character_young_female"
                    )
                    display_name = gr.Textbox(
                        label="Display Name",
                        placeholder="e.g., Deep Narrator, Young Female Character"
                    )
                    description = gr.Textbox(
                        label="Description",
                        placeholder="Character notes, usage instructions...",
                        lines=3
                    )
                    
                    voice_language = gr.Dropdown(
                        choices=list(ChatterboxMultilingualTTS.get_supported_languages().keys()),
                        value="en",
                        label="Language"
                    )
                    
                    voice_ref = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Reference Audio (10-30 seconds recommended)"
                    )
                    
                    with gr.Row():
                        voice_exaggeration = gr.Slider(0.25, 2, value=0.5, label="Exaggeration")
                        voice_cfg = gr.Slider(0.2, 1, value=0.5, label="CFG/Pace")
                        voice_temp = gr.Slider(0.05, 5, value=0.8, label="Temperature")
                    
                    save_voice_btn = gr.Button("💾 Save Voice Profile", variant="primary")
                    save_status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📖 Saved Voices")
                    voice_library_display = gr.HTML(value=refresh_voice_library()[1])
                    refresh_voices_btn = gr.Button("🔄 Refresh Library")
        
        # --- Projects Tab ---
        with gr.TabItem("📁 Projects"):
            gr.Markdown("""
            ## 📚 Audiobook Project Management
            Organize your audiobook projects with character voice assignments.
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ➕ Create New Project")
                    project_name = gr.Textbox(
                        label="Project Name",
                        placeholder="e.g., my_audiobook_2024"
                    )
                    project_desc = gr.Textbox(
                        label="Description",
                        placeholder="Brief description of the audiobook project",
                        lines=2
                    )
                    create_project_btn = gr.Button("📁 Create Project", variant="primary")
                    project_status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column():
                    gr.Markdown("### 📂 Existing Projects")
                    projects_list = gr.Dropdown(
                        choices=[p["name"] for p in get_projects(config.project_path)],
                        label="Select Project"
                    )
                    refresh_projects_btn = gr.Button("🔄 Refresh Projects")

        # --- Script Processing Tab ---
        with gr.TabItem("📖 Script Processing"):
            gr.Markdown("""
            ## 🎭 Multi-Sample Script Processing
            Process audiobook scripts with character voice assignments and batch generation.
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📄 Script Input")
                    script_input = gr.Textbox(
                        label="Audiobook Script",
                        placeholder="Paste your script here with [Speaker] tags...",
                        lines=15,
                        max_lines=20
                    )
                    
                    script_project = gr.Dropdown(
                        choices=[p["name"] for p in get_projects(config.project_path)],
                        label="Project (for character voices)",
                        value=None
                    )
                    
                    with gr.Row():
                        process_script_btn = gr.Button("🔄 Process Script", variant="primary")
                        generate_all_btn = gr.Button("🎙️ Generate All Segments", variant="secondary")
                    
                    process_status = gr.Textbox(label="Processing Status", interactive=False)
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Sample Script")
                    sample_script = gr.Textbox(
                        value=create_sample_script(),
                        label="Example Format",
                        lines=10,
                        interactive=False
                    )
                    
                    load_sample_btn = gr.Button("📋 Load Sample")
            
            with gr.Row():
                processed_segments = gr.Dataframe(
                    headers=["Line", "Speaker", "Text", "Voice", "Status"],
                    datatype=["number", "str", "str", "str", "str"],
                    label="Processed Segments"
                )
            
            with gr.Row():
                batch_audio_output = gr.Audio(label="Generated Audio", type="numpy")

    # --- Event Handlers ---
    save_voice_btn.click(
        fn=save_voice_profile_func,
        inputs=[voice_name, display_name, description, voice_language, 
               voice_exaggeration, voice_cfg, voice_temp, voice_ref],
        outputs=[save_status, voice_library_display]
    )
    
    refresh_voices_btn.click(
        fn=refresh_voice_library,
        outputs=[saved_voices, voice_library_display]
    )
    
    def create_project_func(name, description):
        if not name:
            return "Error: Project name is required!"
        
        success = create_audiobook_project(config.project_path, name, description)
        if success:
            projects = get_projects(config.project_path)
            choices = [p["name"] for p in projects]
            return f"Project '{name}' created successfully!"
        return "Error creating project!"
    
    create_project_btn.click(
        fn=create_project_func,
        inputs=[project_name, project_desc],
        outputs=[project_status]
    )
    
    refresh_projects_btn.click(
        fn=lambda: gr.Dropdown(choices=[p["name"] for p in get_projects(config.project_path)]),
        outputs=[projects_list]
    )

    # Main TTS generation
    run_btn.click(
        fn=generate_tts_audio,
        inputs=[
            text,
            language_id,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
        ],
        outputs=[audio_output],
    )

if __name__ == "__main__":
    demo.launch(mcp_server=True)
