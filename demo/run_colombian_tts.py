#!/usr/bin/env python3
"""
VibeVoice TTS - Voz en Español Colombiano Campesino

Demo preparado con texto en español colombiano auténtico, usando
vocabulario y expresiones del campo colombiano.

Uso:
    python demo/run_colombian_tts.py

Opciones:
    --text_file   Archivo de texto a sintetizar
    --voice       sp-Spk0_woman o sp-Spk1_man
    --output      Archivo de salida
    --device      cuda, cpu, mps
"""

import sys
import os
import argparse
import time
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


def synthesize_colombian(
    model_path: str = "microsoft/VibeVoice-Realtime-0.5B",
    text_file: str = "demo/text_examples/colombian_dialogue.txt",
    output: str = "output_colombian.wav",
    voice: str = "sp-Spk1_man",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    cfg_scale: float = 1.5
):
    print("=" * 60)
    print("VIBEVOICE TTS - ESPAÑOL COLOMBIANO CAMPESINO")
    print("=" * 60)
    print(f"Modelo: {model_path}")
    print(f"Voz: {voice}")
    print(f"Dispositivo: {device}")
    print(f"Texto: {text_file}")
    print()

    from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    if not os.path.exists(text_file):
        print(f"❌ Archivo no encontrado: {text_file}")
        return None

    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        print("❌ El archivo está vacío")
        return None

    text = text.replace("'", "'").replace('"', '"').replace('"', '"')

    print(f"Texto ({len(text)} caracteres):")
    print("-" * 40)
    print(text[:200] + "..." if len(text) > 200 else text)
    print("-" * 40)
    print()

    print("Cargando modelo y processor...")
    start_load = time.time()

    processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)

    dtype = torch.float32 if device in ("cpu", "mps") else torch.bfloat16
    attn = "sdpa" if device in ("cpu", "mps") else "flash_attention_2"

    try:
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
            attn_implementation=attn,
            trust_remote_code=True
        )
        if device == "cpu":
            model = model.to(device)
    except Exception as e:
        print(f"⚠ Fallback a SDPA: {e}")
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
            attn_implementation="sdpa",
            trust_remote_code=True
        )
        if device == "cpu":
            model = model.to(device)

    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)

    print(f"✓ Modelo cargado en {time.time() - start_load:.1f}s")

    voices_dir = os.path.join(os.path.dirname(__file__), "voices", "streaming_model")
    voice_file = os.path.join(voices_dir, f"{voice}.pt")

    if not os.path.exists(voice_file):
        print(f"❌ Voz no encontrada: {voice_file}")
        print("Voces disponibles:")
        for f in sorted(os.listdir(voices_dir)):
            if f.endswith(".pt"):
                print(f"  - {f[:-3]}")
        return None

    print(f"Cargando voz: {voice}")
    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        voice_preset = torch.load(voice_file, map_location=device, weights_only=False)

    print("Procesando texto...")
    inputs = processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=copy.deepcopy(voice_preset),
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    target_device = device if device != "cpu" else "cpu"
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(target_device)

    print(f"Generando audio (cfg_scale={cfg_scale})...")
    start_gen = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            top_p=0.95,
            cfg_scale=cfg_scale,
            max_new_tokens=None,
            tokenizer=processor.tokenizer,
            verbose=True,
            all_prefilled_outputs=copy.deepcopy(voice_preset),
        )

    gen_time = time.time() - start_gen

    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        audio = outputs.speech_outputs[0]
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        sr = 24000
        duration = len(audio) / sr

        print(f"\n⏱ Tiempo: {gen_time:.1f}s")
        print(f"🎵 Duración: {duration:.1f}s")
        print(f"⚡ RTF: {gen_time/duration:.2f}x")

        os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
        processor.save_audio(audio, output_path=output)
        print(f"✓ Guardado: {output}")
        return output
    else:
        print("❌ No se generó audio")
        return None


def main():
    parser = argparse.ArgumentParser(description="VibeVoice TTS - Español Colombiano")
    parser.add_argument("--model_path", type=str, default="microsoft/VibeVoice-Realtime-0.5B")
    parser.add_argument("--text_file", type=str, default="demo/text_examples/colombian_dialogue.txt")
    parser.add_argument("--output", type=str, default="output_colombian.wav")
    parser.add_argument("--voice", type=str, default="sp-Spk1_man",
                        choices=["sp-Spk0_woman", "sp-Spk1_man"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("⚠ Sin GPU NVIDIA - será lento en CPU")

    result = synthesize_colombian(
        model_path=args.model_path,
        text_file=args.text_file,
        output=args.output,
        voice=args.voice,
        device=args.device,
        cfg_scale=args.cfg_scale
    )

    if result:
        print("\n✓ DEMO COMPLETADO")
    else:
        print("\n❌ DEMO FALLÓ")
        sys.exit(1)


if __name__ == "__main__":
    main()