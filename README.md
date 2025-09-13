# 🎧 Chatterbox Multilingual TTS - Audiobook Edition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Languages](https://img.shields.io/badge/languages-23-green.svg)](https://github.com/AmiXDme/Chatterbox-Multilingual-TTS)

**Generate high-quality multilingual speech from text with reference audio styling and audiobook production features.**

This enhanced edition adds powerful voice library management and audiobook project organization to the original Chatterbox Multilingual TTS system.

## ✨ Key Features

### 🌍 **Multilingual Support**
- **23 Languages**: English, French, German, Spanish, Italian, Portuguese, Hindi, Chinese, Japanese, Korean, Arabic, Russian, and more
- **Voice Cloning**: Clone any voice with just 10-30 seconds of reference audio
- **Cross-language**: Use reference audio from one language to generate speech in another

### 📚 **Audiobook Production**
- **Voice Library**: Save and manage character voices with settings
- **Project Management**: Organize audiobook projects with character assignments
- **Unlimited Text**: No character limits for long-form content
- **Batch Processing**: Process entire chapters efficiently

### 🎭 **Voice Management**
- **Voice Profiles**: Save voice settings with names, descriptions, and reference audio
- **Character Consistency**: Use the same voice across entire projects
- **Easy Switching**: Quick dropdown selection between saved voices
- **Settings Persistence**: Store optimized parameters for each character

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/AmiXDme/Chatterbox-Multilingual-TTS.git
cd Chatterbox-Multilingual-TTS

# Install dependencies
pip install -r requirements.txt

# Launch the application
launch_audiobook.bat
```

### 2. First-Time Setup
1. **Launch**: Run `launch_audiobook.bat`
2. **Create Voice**: Go to 📚 Voice Library tab
3. **Upload Audio**: Add 10-30 seconds of reference audio
4. **Save Profile**: Configure and save your first voice
5. **Generate**: Use the voice in TTS Generation tab

### 3. Basic Usage

#### **Single Voice Generation**
```python
from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Initialize model
tts = ChatterboxMultilingualTTS.from_pretrained("cuda")

# Generate speech
audio = tts.generate(
    text="Hello, this is a test of the multilingual TTS system.",
    language_id="en",
    audio_prompt_path="path/to/reference.wav",
    exaggeration=0.5,
    temperature=0.8
)
```

#### **Voice Library Management**
```python
from src.audiobook.voice_management import save_voice_profile

# Save a character voice
save_voice_profile(
    voice_library_path="voice_library",
    voice_name="narrator_deep",
    profile_data={
        "display_name": "Deep Narrator",
        "description": "Professional narrator voice for fantasy audiobooks",
        "language": "en",
        "exaggeration": 0.4,
        "cfg_weight": 0.5,
        "temperature": 0.7
    }
)
```

## 📖 Usage Guide

### **Voice Library Workflow**
1. **📚 Voice Library Tab**: Create and manage voice profiles
2. **🎙️ TTS Generation Tab**: Use saved voices for generation
3. **📁 Projects Tab**: Organize audiobook projects

### **Character Voice Creation**
- **Narrator**: Exaggeration 0.3-0.5, CFG 0.4-0.6
- **Dramatic Characters**: Exaggeration 0.6-0.8
- **Fast Speakers**: CFG 0.6-0.8
- **Slow/Deliberate**: CFG 0.3-0.4

### **Project Organization**
```
audiobook_projects/
├── my_book/
│   ├── project.json
│   ├── chapters/
│   ├── characters/
│   └── output/
voice_library/
├── narrator_male/
├── character_female/
└── villain_gravelly/
```

## 🛠️ Technical Details

### **Supported Languages**
| Language | Code | Language | Code |
|----------|------|----------|------|
| Arabic | ar | Japanese | ja |
| Chinese | zh | Korean | ko |
| Danish | da | Malay | ms |
| Dutch | nl | Norwegian | no |
| English | en | Polish | pl |
| Finnish | fi | Portuguese | pt |
| French | fr | Russian | ru |
| German | de | Spanish | es |
| Greek | el | Swedish | sv |
| Hebrew | he | Swahili | sw |
| Hindi | hi | Turkish | tr |
| Italian | it | | |

### **System Requirements**
- **Python**: 3.10+
- **PyTorch**: 2.4.1+
- **CUDA**: Optional (works on CPU)
- **RAM**: 8GB+ recommended
- **Storage**: 2GB+ for models

### **Dependencies**
```
torch==2.4.1
torchaudio==2.4.1
gradio
librosa==0.10.0
transformers==4.46.3
numpy==1.26.0
```

## 📊 Performance Notes

- **GPU**: Significantly faster processing (recommended)
- **CPU**: Functional but slower
- **Memory**: Models require ~2GB VRAM
- **Text Length**: No artificial limits
- **Audio Quality**: 24kHz sample rate

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Resemble AI** for the original Chatterbox Multilingual TTS
- **Hugging Face** for transformer models
- **Gradio** for the web interface

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/AmiXDme/Chatterbox-Multilingual-TTS/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AmiXDme/Chatterbox-Multilingual-TTS/discussions)
- **Documentation**: This README + inline code comments

---

**Ready to create amazing multilingual audiobooks?** 🎧✨

Launch with: `launch_audiobook.bat`