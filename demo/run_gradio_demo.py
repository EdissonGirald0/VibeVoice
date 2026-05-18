#!/usr/bin/env python3
"""
Demo 2: VibeVoice ASR - Interfaz Web con Gradio

Este demo запускает una interfaz web interactiva para transcribir audio
usando Gradio. Perfecto para probar con diferentes archivos de audio.

Uso:
    python demo/run_gradio_demo.py --share

Requisitos:
    - Gradio instalado
    - GPU NVIDIA recomendada (funciona en CPU pero es lento)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gradio as gr
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


class VibeVoiceASRGradioDemo:
    def __init__(self, model_path: str = "microsoft/VibeVoice-ASR"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.ready = False

    def load(self):
        print("=" * 60)
        print("CARGANDO MODELO VIBEVOICE-ASR")
        print("=" * 60)

        self.processor = VibeVoiceASRProcessor.from_pretrained(
            self.model_path,
            language_model_pretrained_name="Qwen/Qwen2.5-7B"
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True
        )
        self.model.eval()

        print("✓ Modelo cargado y listo")
        self.ready = True
        return self

    def transcribe(self, audio_file, language=None, max_new_tokens=32768):
        if audio_file is None:
            return "Por favor suba un archivo de audio"

        try:
            inputs = self.processor(
                audio=audio_file,
                sampling_rate=None,
                return_tensors="pt",
                add_generation_prompt=True
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                          for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.processor.pad_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

            generated_text = self.processor.decode(
                output_ids[0, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            try:
                segments = self.processor.post_process_transcription(generated_text)
                formatted = self._format_segments(segments)
                return formatted
            except:
                return generated_text

        except Exception as e:
            return f"Error: {str(e)}"

    def _format_segments(self, segments):
        if not segments:
            return "No se detectaron segmentos"
        result = []
        for seg in segments[:50]:
            start = seg.get("start_time", "N/A")
            end = seg.get("end_time", "N/A")
            speaker = seg.get("speaker_id", "N/A")
            text = seg.get("text", "")
            result.append(f"[{start} - {end}] Speaker {speaker}: {text}")
        return "\n".join(result)


def main():
    print("=" * 60)
    print("VIBEVOICE ASR - GRADIO DEMO")
    print("=" * 60)

    demo = VibeVoiceASRGradioDemo()
    demo.load()

    interface = gr.Interface(
        fn=demo.transcribe,
        title="🎙️ VibeVoice ASR - Transcriptor de Audio",
        description="""
        ## VibeVoice ASR
        Transcriptor de audio de larga duración con speaker diarization.

        **Características:**
        - Audio hasta 60 minutos
        - Speaker diarization
        - Timestamps por segmento
        - Soporta +50 idiomas
        """,
        inputs=[
            gr.Audio(label="Archivo de Audio", type="filepath"),
            gr.Slider(1024, 32768, value=32768, label="Max tokens"),
        ],
        outputs=gr.Textbox(label="Transcripción", lines=20),
        examples=[
            ["demo/asr_demo/demo1-chat.mp3"],
            ["demo/asr_demo/demo2-song.mp3"],
            ["demo/asr_demo/demo3-hotwords.wav"],
        ],
        theme=gr.themes.Soft()
    )

    print("\n🌐 Iniciando interfaz Gradio...")
    print("   URL local: http://localhost:7860")
    print("   Presiona Ctrl+C para detener\n")

    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)


if __name__ == "__main__":
    main()