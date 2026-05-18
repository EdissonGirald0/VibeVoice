# VibeVoice - Demos Preparados

Directorio con scripts preparados para ejecutar demos de VibeVoice.

## Estructura

```
demo/
├── check_requirements.py      # Verificador de sistema
├── run_asr_demo.py            # Demo 1: ASR (transcripción)
├── run_gradio_demo.py         # Demo 2: Interfaz web Gradio
├── run_realtime_tts_demo.py   # Demo 3: TTS en tiempo real
├── run_all_demos.py           # Script maestro
├── DEMOS.md                   # Este archivo
└── asr_demo/                  # Archivos de audio para pruebas
    ├── demo1-chat.mp3
    ├── demo2-song.mp3
    └── demo3-hotwords.wav
```

## Uso Rápido

### 1. Verificar Sistema

```bash
python demo/check_requirements.py
```

### 2. Ejecutar Todas las Demos

```bash
python demo/run_all_demos.py
```

### 3. Ejecutar Demos Individuales

**Demo ASR (Transcripción de Audio):**
```bash
python demo/run_asr_demo.py --audio_file demo/asr_demo/demo1-chat.mp3
```

**Demo Gradio (Interfaz Web):**
```bash
python demo/run_gradio_demo.py --share
```
Luego abrir: http://localhost:7860

**Demo TTS (Síntesis de Voz):**
```bash
python demo/run_realtime_tts_demo.py --text "Hola, esto es una prueba"
```

## Descripción de Demos

### Demo 1: ASR (Automatic Speech Recognition)

**Script:** `run_asr_demo.py`

Transcribe archivos de audio a texto con:
- Audio hasta 60 minutos
- Speaker diarization (quién habló)
- Timestamps por segmento
- Soporte para +50 idiomas

**Opciones:**
```bash
--model_path    # Ruta del modelo (default: microsoft/VibeVoice-ASR)
--audio_file    # Archivo de audio a transcribir
--device        # cuda, cpu, mps (default: auto-detecta)
```

### Demo 2: Gradio (Interfaz Web)

**Script:** `run_gradio_demo.py`

Interfaz web interactiva para transcribir audio.

**Características:**
- Subir archivos de audio
- Visualizar transcripción
- Ejemplos pre-cargados

**Uso:**
```bash
python demo/run_gradio_demo.py --share
```

### Demo 3: TTS Realtime (Síntesis de Voz)

**Script:** `run_realtime_tts_demo.py`

Genera audio desde texto con:
- Latencia ~200-300ms
- Streaming en tiempo real
- Múltiples voces

**Opciones:**
```bash
--text          # Texto a sintetizar
--text_file     # Archivo de texto (.txt)
--output        # Archivo de salida (default: output_tts.wav)
--voice         # Nombre de la voz (default: Wayne)
--device        # cuda, cpu, mps
```

## Requisitos

- Python 3.10+
- PyTorch 2.0+
- Transformers >=4.51.3
- 8GB+ RAM (16GB+ para modelos de 7B)
- GPU NVIDIA recomendada (funciona en CPU pero es lento)

## Notas

- La primera ejecución descargará modelos de HuggingFace (~14GB)
- Sin GPU, el procesamiento es significativamente más lento
- Los archivos de audio de demo están en `demo/asr_demo/`