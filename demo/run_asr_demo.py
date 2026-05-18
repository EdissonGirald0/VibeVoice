#!/usr/bin/env python3
"""
Demo 1: VibeVoice ASR - Transcripción de Audio a Texto

Este demo muestra cómo usar VibeVoice-ASR para transcribir archivos de audio
de hasta 60 minutos en una sola pasada, con speaker diarization y timestamps.

Uso:
    python demo/run_asr_demo.py

Requisitos:
    - GPU NVIDIA con 16GB+ VRAM (o CPU con paciencia)
    - Archivos de audio en demo/asr_demo/
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


class VibeVoiceASRDemmo:
    def __init__(self, model_path: str = "microsoft/VibeVoice-ASR", device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        print("=" * 60)
        print("CARGANDO MODELO VIBEVOICE-ASR")
        print("=" * 60)
        print(f"Modelo: {self.model_path}")
        print(f"Dispositivo: {self.device}")
        print()

        start = time.time()

        print("Cargando processor...")
        self.processor = VibeVoiceASRProcessor.from_pretrained(
            self.model_path,
            language_model_pretrained_name="Qwen/Qwen2.5-7B"
        )

        print("Cargando modelo...")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map=self.device if self.device == "auto" else None,
            attn_implementation="sdpa",
            trust_remote_code=True
        )

        if self.device != "auto":
            self.model = self.model.to(self.device)

        self.model.eval()

        print(f"\n✓ Modelo cargado en {time.time() - start:.1f}s")
        return self

    def transcribe(self, audio_path: str) -> dict:
        print(f"\n{'=' * 60}")
        print(f"TRANSCRIBIENDO: {os.path.basename(audio_path)}")
        print("=" * 60)

        start = time.time()

        inputs = self.processor(
            audio=audio_path,
            sampling_rate=None,
            return_tensors="pt",
            add_generation_prompt=True
        )

        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        print(f"  Input shape: {inputs['input_ids'].shape}")
        print(f"  Audio tokens: {inputs['speech_tensors'].shape[1] if 'speech_tensors' in inputs else 'N/A'}")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=32768,
                pad_token_id=self.processor.pad_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        generated_text = self.processor.decode(
            output_ids[0, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        elapsed = time.time() - start

        try:
            segments = self.processor.post_process_transcription(generated_text)
        except:
            segments = []

        result = {
            "file": os.path.basename(audio_path),
            "text": generated_text,
            "segments": segments,
            "time": elapsed
        }

        print(f"\n⏱ Tiempo de procesamiento: {elapsed:.1f}s")
        print(f"  Segmentos detectados: {len(segments)}")

        return result

    def print_result(self, result: dict):
        print(f"\n{'=' * 60}")
        print("RESULTADO DE TRANSCRIPCIÓN")
        print("=" * 60)

        print(f"\n📄 Texto completo:")
        print("-" * 40)
        print(result["text"][:2000] + ("..." if len(result["text"]) > 2000 else ""))

        if result["segments"]:
            print(f"\n👥 Segmentos (primeras 10 líneas):")
            print("-" * 40)
            for i, seg in enumerate(result["segments"][:10]):
                start = seg.get("start_time", "N/A")
                end = seg.get("end_time", "N/A")
                speaker = seg.get("speaker_id", "N/A")
                text = seg.get("text", "")[:100]
                print(f"  [{start} - {end}] Speaker {speaker}: {text}...")
            if len(result["segments"]) > 10:
                print(f"\n  ... y {len(result['segments']) - 10} segmentos más")


def main():
    parser = argparse.ArgumentParser(description="VibeVoice ASR Demo")
    parser.add_argument("--model_path", type=str, default="microsoft/VibeVoice-ASR")
    parser.add_argument("--audio_file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    demo = VibeVoiceASRDemmo(model_path=args.model_path, device=args.device)
    demo.load()

    demo_dir = os.path.join(os.path.dirname(__file__), "asr_demo")

    if args.audio_file:
        audio_files = [args.audio_file]
    else:
        audio_files = [os.path.join(demo_dir, f) for f in os.listdir(demo_dir)
                       if f.endswith(('.mp3', '.mp4', '.wav'))]

    for audio_path in audio_files:
        if os.path.exists(audio_path):
            result = demo.transcribe(audio_path)
            demo.print_result(result)

            save_path = audio_path + ".transcription.txt"
            with open(save_path, "w") as f:
                f.write(result["text"])
            print(f"\n💾 Transcripción guardada en: {save_path}")
        else:
            print(f"⚠ Archivo no encontrado: {audio_path}")

    print("\n" + "=" * 60)
    print("✓ DEMO ASR COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()