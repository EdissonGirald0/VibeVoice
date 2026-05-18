#!/usr/bin/env python3
"""
Demo 3: VibeVoice Realtime TTS - Síntesis de Voz en Streaming

Este demo muestra cómo usar VibeVoice-Realtime para sintetizar voz
a partir de texto con latencia ultra-baja (~200-300ms).

Uso:
    python demo/run_realtime_tts_demo.py --text "Hola mundo"

Requisitos:
    - GPU recomendada (funciona en CPU pero es lento)
    - Archivos de voz en demo/voices/
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor


class VibeVoiceRealtimeTTSDemo:
    def __init__(self, model_path: str = "microsoft/VibeVoice-Realtime-0.5B", device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        print("=" * 60)
        print("CARGANDO MODELO VIBEVOICE-REALTIME")
        print("=" * 60)
        print(f"Modelo: {self.model_path}")
        print(f"Dispositivo: {self.device}")
        print()

        start = time.time()

        print("Cargando processor...")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        print("Cargando modelo...")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map=self.device,
            trust_remote_code=True
        )

        if self.device != "auto":
            self.model = self.model.to(self.device)

        self.model.eval()

        print(f"\n✓ Modelo cargado en {time.time() - start:.1f}s")
        return self

    def synthesize(self, text: str, output_path: str = "output.wav", voice: str = "Wayne"):
        print(f"\n{'=' * 60}")
        print(f"SINTETIZANDO TEXTO")
        print("=" * 60)
        print(f"Texto: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"Voz: {voice}")
        print()

        start = time.time()

        try:
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                add_generation_prompt=True
            )

            if self.device != "cpu":
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                          for k, v in inputs.items()}

            print("Generando audio (streaming)...")

            import numpy as np
            all_audio = []

            with torch.no_grad():
                for chunk in self.model.stream_generate(**inputs):
                    if isinstance(chunk, torch.Tensor):
                        chunk = chunk.cpu().numpy()
                    all_audio.append(chunk)

            if all_audio:
                audio = np.concatenate(all_audio)
            else:
                print("⚠ No se generó audio")
                return None

            try:
                import scipy.io.wavfile as wavfile
                wavfile.write(output_path, 24000, audio)
                elapsed = time.time() - start
                print(f"\n✓ Audio guardado en: {output_path}")
                print(f"⏱ Tiempo: {elapsed:.1f}s")
                print(f"🎵 Duración: {len(audio)/24000:.1f}s")
                return output_path
            except Exception as e:
                print(f"⚠ Error guardando WAV: {e}")
                return audio

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    parser = argparse.ArgumentParser(description="VibeVoice Realtime TTS Demo")
    parser.add_argument("--model_path", type=str, default="microsoft/VibeVoice-Realtime-0.5B")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--text_file", type=str, default=None)
    parser.add_argument("--output", type=str, default="output_tts.wav")
    parser.add_argument("--voice", type=str, default="Wayne")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("⚠ AVISO: No se detectó GPU NVIDIA. El proceso será lento en CPU.")

    demo = VibeVoiceRealtimeTTSDemo(model_path=args.model_path, device=args.device)
    demo.load()

    if args.text_file and os.path.exists(args.text_file):
        with open(args.text_file, "r") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        text = "Hola, esto es una prueba del sistema de síntesis de voz VibeVoice. Este modelo puede generar audio de alta calidad en tiempo real."

    demo.synthesize(text, output_path=args.output, voice=args.voice)

    print("\n" + "=" * 60)
    print("✓ DEMO TTS COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()