# VibeVoice - Análisis y Guía de Demo

## Resumen

**VibeVoice** es un framework de Voice AI de código abierto desarrollado por Microsoft que proporciona:

| Modelo | Parámetros | Función | Estado |
|--------|------------|---------|--------|
| VibeVoice-ASR | 7B | Transcripción de audio hasta 60 min | ✅ Activo |
| VibeVoice-TTS | 1.5B | Síntesis de voz hasta 90 min | ⚠️ Deshabilitado |
| VibeVoice-Realtime | 0.5B | Streaming TTS en tiempo real | ✅ Activo |

## Tecnologías Principales

- **Backend**: PyTorch + Transformers >=4.51.3
- **LLM Base**: Qwen2.5 (1.5B y 7B)
- **Procesamiento de Audio**: librosa, pydub, av (FFmpeg)
- **Inference**: vLLM (plugin nativo), FastAPI, Gradio
- **Streaming**: WebSocket + aiortc (WebRTC)

## Arquitectura

```
Audio Input
    │
    ▼
┌─────────────────┐
│    Processor    │  (tokenization, normalization)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Model (LLM +   │
│  Tokenizers +   │
│  Diffusion Head)│
└─────────────────┘
    │
    ▼
Audio/Text Output
```

---

## Instalación

```bash
# Clonar repositorio
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice

# Instalar como paquete
pip install -e .

# Para streaming TTS
pip install -e .[streamingtts]
```

---

## Demos Disponibles

### 1. ASR (Transcripción) con Gradio

```bash
python demo/vibevoice_asr_gradio_demo.py --model_path microsoft/VibeVoice-ASR --share
```

**Características:**
- Carga archivos de audio
- Soporta +50 idiomas
- Muestra timestamps y speaker diarization
- Interfaz web interactiva

### 2. ASR desde archivo (script directo)

```bash
python demo/vibevoice_asr_inference_from_file.py \
    --model_path microsoft/VibeVoice-ASR \
    --audio_file demo/asr_demo/sample.wav
```

### 3. Realtime TTS Demo

```bash
python demo/vibevoice_realtime_demo.py --model_path microsoft/VibeVoice-Realtime-0.5B
```

**Características:**
- Latencia ~200-300ms
- Streaming de texto
- English only

### 4. Servidor vLLM (API OpenAI-compatible)

```bash
python vllm_plugin/scripts/start_server.py --dp 4
```

**Endpoints:**
- `POST /v1/chat/completions` - Transcripción
- `GET /health` - Health check

---

## Uso como Librería Python

### ASR - Transcripción de Audio

```python
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

# Cargar modelo y procesador
model = VibeVoiceASRForConditionalGeneration.from_pretrained("microsoft/VibeVoice-ASR")
processor = VibeVoiceASRProcessor.from_pretrained("microsoft/VibeVoice-ASR")

# Transcribir audio
inputs = processor(audio, return_tensors="pt")
outputs = model.generate(**inputs)
transcription = processor.batch_decode(outputs)[0]

print(transcription)
# {"text": "...", "segments": [...]}
```

### Realtime TTS - Síntesis en Streaming

```python
from vibevoice.modular.modeling_vibevoice_streaming import VibeVoiceStreamingForConditionalGeneration
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

# Cargar modelo
model = VibeVoiceStreamingForConditionalGeneration.from_pretrained("microsoft/VibeVoice-Realtime-0.5B")
processor = VibeVoiceStreamingProcessor.from_pretrained("microsoft/VibeVoice-Realtime-0.5B")

# Streaming TTS
text = "Hello, this is a test of the VibeVoice streaming TTS system."
inputs = processor(text=text, return_tensors="pt")

# Generar y streamear audio chunk por chunk
for audio_chunk in model.stream_generate(**inputs):
    play_audio(audio_chunk)
```

---

## Fine-tuning con LoRA

```bash
torchrun --nproc_per_node=1 finetuning-asr/lora_finetune.py \
    --model_path microsoft/VibeVoice-ASR \
    --data_dir ./dataset \
    --output_dir ./output \
    --num_train_epochs 3
```

---

## Integración con nuestro Stack

### Como Microservicio (Recomendado)

```
┌─────────────────────────────────────────┐
│            Tu Aplicación                │
│   (Chatbot, Assistant, Analytics)       │
└─────────────────┬───────────────────────┘
                  │ HTTP/WebSocket
                  ▼
┌─────────────────────────────────────────┐
│          VibeVoice API                  │
│     (vLLM + FastAPI + Gradio)          │
│   Puerto: 8000 (API), 7860 (Gradio)    │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│     GPU Server (NVIDIA)                 │
│   (VibeVoice-ASR o Realtime)           │
└─────────────────────────────────────────┘
```

### Puntos de Integración

| Componente | Integración |
|------------|-------------|
| REST API | `POST /v1/chat/completions` |
| WebSocket | `demo/web/app.py` para streaming |
| Python Lib | Import directo `from vibevoice...` |
| vLLM | Plugin nativo con entry points |

---

## Casos de Uso

1. **Call Center Analytics** - Transcripción + análisis de sentimiento
2. **Voice Assistants** - Integración con sistemas RAG/LangChain
3. **Accessibility** - Transcripción en tiempo real
4. **Content Creation** - Generación de audio para videos

---

## Consideraciones de Seguridad

- TTS completo fue deshabilitado por riesgos de deepfakes
- ASR requiere disclosure de contenido AI
- Evaluar antes de producción

---

## Recursos

- Repo: https://github.com/microsoft/VibeVoice
- Docs: `docs/` en el repositorio
- Modelos: HuggingFace `microsoft/VibeVoice-ASR`, `microsoft/VibeVoice-Realtime-0.5B`