# 🎙️ VibeVoice Audio Studio

Sistema de generación de audio de alta calidad en español colombiano para videos.

## Descripción

Pipeline completo para generar audios con vocabulario colombiano auténtico usando **VibeVoice-Realtime** de Microsoft, con post-procesamiento profesional para producción de video.

## Estructura del Proyecto

```
VibeVoice/
├── audio_studio.py           # Interfaz Gradio (GUI)
├── generate_tts.py           # Generación TTS
├── post_process.py          # Post-procesamiento de audio
├── make_audio.sh            # Script shell workflow completo
├── configs/
│   └── generate_config.json # Configuración default
├── colombian_library/        # Biblioteca de textos
│   ├── salud/               # Textos de salud y remedios
│   │   ├── introduccion.txt
│   │   ├── plantas_basicas.txt
│   │   └── preparaciones.txt
│   └── campo/               # Textos de vida rural
│       ├── aprendizajes.txt
│       └── sabiduria_rural.txt
├── demo/                     # Demos originales de VibeVoice
└── .venv/                   # Entorno virtual Python
```

## Requisitos

- Python 3.10+
- FFmpeg instalado
- 8GB+ RAM (16GB+ recomendado)
- GPU NVIDIA (opcional, funciona en CPU)

## Instalación

```bash
# Crear entorno virtual
python3 -m venv .venv

# Activar entorno
source .venv/bin/activate

# Instalar dependencias
pip install torch transformers gradio scipy soundfile pydub librosa

# Verificar FFmpeg
ffmpeg -version
```

## Uso

### Interfaz Gradio (Recomendado)

```bash
python audio_studio.py
# Abrir: http://localhost:7860
```

### Script Shell

```bash
# Texto directo → MP3
./make_audio.sh "Buenos días mijo" audio.mp3

# Batch desde directorio
./make_audio.sh --batch --dir textos/ --output audios/
```

### Python Scripts

```bash
# Generar desde texto
python generate_tts.py --text "Hola mijo" --output audio.wav

# Batch
python generate_tts.py --batch --dir textos/ --output output/

# Post-procesar
python post_process.py input.wav output.mp3
```

## Configuración

### Parámetros de Generación

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `voice` | sp-Spk1_man | Voz masculina española |
| `cfg_scale` | 1.5 | CFG scale (1.0-2.0) |
| `ddpm_steps` | 5 | Pasos de diffusion (3-10) |
| `device` | cpu | cpu, cuda, mps |

### Voces Disponibles

- `sp-Spk1_man` - Masculina española (default)
- `sp-Spk0_woman` - Femenina española

### Post-procesamiento

- Normalización LUFS: -16 (estándar podcast)
- Compresión: ratio 3:1
- Silencios removidos
- Fade in/out

## Biblioteca de Textos

Textos de ejemplo en español colombiano auténtico:

### Salud y Remedios

- **introduccion.txt**: Introducción general a plantas medicinales
- **plantas_basicas.txt**: Descripción de plantas comunes (ajenjo, manzanilla, ispagulla, menta, valeriana)
- **preparaciones.txt**: Cómo preparar infusiones y remedios

### Campo y Vida Rural

- **aprendizajes.txt**: Lecciones de vida del campo
- **sabiduria_rural.txt**: Refranes y sabiduría popular

## Workflow Completo

```
Texto → generate_tts.py → WAV → post_process.py → MP3 (listo para video)
         (CPU)            (FFmpeg)           (192kbps, -16 LUFS)
```

## Optimización CPU

Sin GPU NVIDIA, el proceso es más lento:
- Texto corto (<50 palabras): ~15-30 segundos
- Texto medio (50-200 palabras): ~1-3 minutos
- Texto largo (>200 palabras): ~5-10 minutos

## Fine-tuning (Próxima Etapa)

Para crear voz colombiana auténtica, se requiere:
- Dataset: 30-60 min de audio colombiano
- GPU: 16GB+ VRAM (NVIDIA T4 o similar)
- Tiempo: ~8-12 horas de entrenamiento

## Troubleshooting

### Error: FFmpeg no encontrado
```bash
# Linux
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### Error: Memoria insuficiente
Reducir `ddpm_steps` a 3 o 4

### Audio con静音
El post-procesamiento ya elimina silencios largos

## Licencia

VibeVoice es un proyecto de Microsoft. Ver LICENSE en el repositorio original.

## Créditos

- **VibeVoice**: https://github.com/microsoft/VibeVoice
- **Modelo**: microsoft/VibeVoice-Realtime-0.5B (HuggingFace)