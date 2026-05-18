#!/usr/bin/env python3
"""
VibeVoice TTS Generator - Audio de Alta Calidad para Videos

Script para generar audio en español colombiano usando VibeVoice-Realtime
con parámetros optimizados para CPU y post-procesamiento profesional.

Uso:
    python generate_tts.py --text "Hola mijo, vamos a hablar de..."
    python generate_tts.py --file texto.txt --output audio.wav
    python generate_tts.py --batch --dir textos/ --output output/
"""

import os
import sys
import argparse
import time
import json
import copy
import glob
from pathlib import Path

import torch
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast


DEFAULT_CONFIG = {
    "model": "microsoft/VibeVoice-Realtime-0.5B",
    "voice": "sp-Spk1_man",
    "cfg_scale": 1.5,
    "ddpm_steps": 5,
    "device": "cpu",
    "dtype": "float32",
    "attn_implementation": "sdpa"
}


class VibeVoiceGenerator:
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG.copy()
        self.model = None
        self.processor = None
        self.voice_preset = None
        self.voice_dir = None

    def load(self):
        """Cargar modelo y processor"""
        print("=" * 60)
        print("CARGANDO MODELO VIBEVOICE-REALTIME")
        print("=" * 60)
        print(f"Modelo: {self.config['model']}")
        print(f"Voz: {self.config['voice']}")
        print(f"Dispositivo: {self.config['device']}")
        print(f"CFG Scale: {self.config['cfg_scale']}")
        print(f"DDPM Steps: {self.config['ddpm_steps']}")
        print()

        start = time.time()

        print("Cargando processor...")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(
            self.config["model"]
        )

        print("Cargando modelo...")
        dtype = torch.float32 if self.config["device"] == "cpu" else torch.bfloat16

        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.config["model"],
                torch_dtype=dtype,
                device_map=self.config["device"] if self.config["device"] != "cpu" else None,
                attn_implementation=self.config["attn_implementation"],
                trust_remote_code=True
            )
            if self.config["device"] == "cpu":
                self.model = self.model.to("cpu")
        except Exception as e:
            print(f"⚠ Fallback a SDPA: {e}")
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.config["model"],
                torch_dtype=dtype,
                device_map=self.config["device"] if self.config["device"] != "cpu" else None,
                attn_implementation="sdpa",
                trust_remote_code=True
            )
            if self.config["device"] == "cpu":
                self.model = self.model.to("cpu")

        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=self.config["ddpm_steps"])

        self._load_voice()

        print(f"\n✓ Modelo cargado en {time.time() - start:.1f}s")
        return self

    def _load_voice(self):
        """Cargar preset de voz"""
        self.voice_dir = os.path.join(
            os.path.dirname(__file__),
            "demo", "voices", "streaming_model"
        )
        voice_file = os.path.join(self.voice_dir, f"{self.config['voice']}.pt")

        if not os.path.exists(voice_file):
            available = os.listdir(self.voice_dir) if os.path.exists(self.voice_dir) else []
            raise FileNotFoundError(
                f"Voz {self.config['voice']} no encontrada en {self.voice_dir}\n"
                f"Voces disponibles: {[f[:-3] for f in available if f.endswith('.pt')]}"
            )

        print(f"Cargando voz: {self.config['voice']}")
        with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
            self.voice_preset = torch.load(voice_file, map_location=self.config["device"], weights_only=False)

    def generate(self, text, output_path="output.wav", verbose=True):
        """Generar audio desde texto"""
        if not text.strip():
            raise ValueError("El texto está vacío")

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"GENERANDO AUDIO")
            print("=" * 60)
            print(f"Texto: {text[:100]}{'...' if len(text) > 100 else ''}")

        start = time.time()

        inputs = self.processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=copy.deepcopy(self.voice_preset),
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        device = self.config["device"] if self.config["device"] != "cpu" else "cpu"
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device)

        if verbose:
            print("Generando...")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                top_p=0.95,
                cfg_scale=self.config["cfg_scale"],
                max_new_tokens=None,
                tokenizer=self.processor.tokenizer,
                verbose=verbose,
                all_prefilled_outputs=copy.deepcopy(self.voice_preset),
            )

        gen_time = time.time() - start

        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            audio = outputs.speech_outputs[0]
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()

            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

            self.processor.save_audio(audio, output_path=output_path)

            duration = len(audio) / 24000
            rtf = gen_time / duration if duration > 0 else 0

            if verbose:
                print(f"\n✓ Audio guardado: {output_path}")
                print(f"⏱ Tiempo: {gen_time:.1f}s")
                print(f"🎵 Duración: {duration:.1f}s")
                print(f"⚡ RTF: {rtf:.2f}x")

            return {
                "output_path": output_path,
                "duration": duration,
                "generation_time": gen_time,
                "rtf": rtf
            }
        else:
            raise RuntimeError("No se generó audio")

    def generate_from_file(self, input_file, output_dir=None):
        """Generar audio desde archivo de texto"""
        if output_dir is None:
            output_dir = os.path.dirname(input_file) or "."

        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            raise ValueError(f"Archivo vacío: {input_file}")

        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_path = os.path.join(output_dir, f"{base_name}.wav")

        return self.generate(text, output_path)

    def generate_batch(self, input_files, output_dir=".", show_progress=True):
        """Generar audio desde múltiples archivos"""
        results = []
        total = len(input_files)

        for i, input_file in enumerate(input_files):
            if show_progress:
                print(f"\n[{i+1}/{total}] Procesando: {input_file}")

            try:
                result = self.generate_from_file(input_file, output_dir)
                results.append({"file": input_file, "status": "success", **result})
                if show_progress:
                    print(f"✓ OK: {result['output_path']} ({result['duration']:.1f}s)")
            except Exception as e:
                results.append({"file": input_file, "status": "error", "error": str(e)})
                if show_progress:
                    print(f"✗ ERROR: {e}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="VibeVoice TTS Generator - Audio de Alta Calidad",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Texto único
  python generate_tts.py --text "Hola mijo, vamos a hablar de plantas medicinales"

  # Desde archivo
  python generate_tts.py --file texto.txt --output audio.wav

  # Batch desde directorio
  python generate_tts.py --batch --dir textos/ --output output/

  # Con configuración custom
  python generate_tts.py --text "..." --cfg_scale 1.4 --ddpm_steps 4
        """
    )

    parser.add_argument("--text", type=str, help="Texto a sintetizar")
    parser.add_argument("--file", type=str, help="Archivo de texto de entrada")
    parser.add_argument("--batch", action="store_true", help="Modo batch")
    parser.add_argument("--dir", type=str, help="Directorio con archivos de texto")
    parser.add_argument("--output", type=str, default="output.wav", help="Archivo o directorio de salida")
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG["model"], help="Modelo a usar")
    parser.add_argument("--voice", type=str, default=DEFAULT_CONFIG["voice"], help="Voz a usar (sp-Spk0_woman, sp-Spk1_man)")
    parser.add_argument("--cfg_scale", type=float, default=DEFAULT_CONFIG["cfg_scale"], help="CFG scale (default: 1.5)")
    parser.add_argument("--ddpm_steps", type=int, default=DEFAULT_CONFIG["ddpm_steps"], help="DDPM steps (default: 5)")
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"], help="Device (cpu, cuda, mps)")
    parser.add_argument("--config", type=str, help="Archivo de configuración JSON")

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config.update(json.load(f))

    for key in ["model", "voice", "cfg_scale", "ddpm_steps", "device"]:
        attr = getattr(args, key, None)
        if attr is not None:
            config[key] = attr

    generator = VibeVoiceGenerator(config)

    if args.text:
        output = args.output if args.output else "output.wav"
        generator.load()
        result = generator.generate(args.text, output)
        print(f"\n✓ Generado: {result['output_path']} ({result['duration']:.1f}s)")

    elif args.file:
        generator.load()
        result = generator.generate_from_file(args.file, os.path.dirname(args.output) or ".")
        print(f"\n✓ Generado: {result['output_path']} ({result['duration']:.1f}s)")

    elif args.batch and args.dir:
        files = []
        for ext in ["*.txt", "*.text"]:
            files.extend(glob.glob(os.path.join(args.dir, ext)))

        if not files:
            print(f"No se encontraron archivos .txt en {args.dir}")
            return 1

        print(f"Encontrados {len(files)} archivos")
        generator.load()
        results = generator.generate_batch(files, args.output or ".")

        success = sum(1 for r in results if r["status"] == "success")
        errors = len(results) - success
        print(f"\n{'=' * 60}")
        print(f"RESULTADOS: {success} OK, {errors} errores")
        print("=" * 60)

        for r in results:
            if r["status"] == "error":
                print(f"✗ {r['file']}: {r['error']}")

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)