#!/usr/bin/env python3
"""
VibeVoice Audio Studio - Interfaz Gráfica para Crear Audios

Interfaz visual para generar audio de alta calidad con vocabulario
colombiano auténtico usando VibeVoice-Realtime.

Uso:
    python audio_studio.py
    python audio_studio.py --share  # Con URL pública
"""

import os
import sys
import shutil
import tempfile
import threading
from pathlib import Path

import gradio as gr
import torch

from generate_tts import VibeVoiceGenerator, DEFAULT_CONFIG
from post_process import AudioPostProcessor


class AudioStudio:
    def __init__(self):
        self.generator = None
        self.processor = AudioPostProcessor()
        self.config = DEFAULT_CONFIG.copy()
        self.model_loaded = False

    def load_model(self, progress=gr.Progress):
        """Cargar modelo VibeVoice"""
        if self.model_loaded:
            return "✓ Modelo ya cargado"

        progress(0, desc="Cargando modelo...")
        self.generator = VibeVoiceGenerator(self.config)

        progress(0.3, desc="Cargando processor...")
        try:
            self.generator.load()
            self.model_loaded = True
            return "✓ Modelo cargado exitosamente"
        except Exception as e:
            return f"✗ Error: {str(e)}"

    def generate_audio(self, text, voice, cfg_scale, ddpm_steps, lufs_target, progress=gr.Progress):
        """Generar audio desde texto"""
        if not self.model_loaded:
            return None, "Primero debe cargar el modelo"

        if not text.strip():
            return None, "Ingrese texto para generar"

        progress(0, desc="Generando audio...")

        self.config["voice"] = voice
        self.config["cfg_scale"] = cfg_scale
        self.config["ddpm_steps"] = ddpm_steps
        self.generator.config = self.config

        temp_wav = tempfile.mktemp(suffix=".wav")

        try:
            progress(0.2, desc="Generando...")
            result = self.generator.generate(text, temp_wav, verbose=False)

            progress(0.7, desc="Post-procesando...")
            temp_mp3 = tempfile.mktemp(suffix=".mp3")
            self.processor.config["lufs_target"] = lufs_target
            self.processor.process(temp_wav, temp_mp3, verbose=False)

            progress(1.0, desc="Listo!")
            return temp_mp3, f"✓ Audio generado: {result['duration']:.1f}s"

        except Exception as e:
            return None, f"✗ Error: {str(e)}"
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except:
                    pass

    def generate_batch(self, files, voice, cfg_scale, ddpm_steps, lufs_target, progress=gr.Progress):
        """Generar múltiples audios"""
        if not self.model_loaded:
            return "Primero debe cargar el modelo"

        if not files:
            return "Seleccione archivos de texto"

        results = []
        success = 0
        errors = 0

        self.config["voice"] = voice
        self.config["cfg_scale"] = cfg_scale
        self.config["ddpm_steps"] = ddpm_steps
        self.generator.config = self.config

        self.processor.config["lufs_target"] = lufs_target

        for i, file in enumerate(files):
            progress((i + 1) / len(files), desc=f"Procesando {i+1}/{len(files)}")

            try:
                with open(file.name, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                if not text:
                    errors += 1
                    results.append(f"✗ {os.path.basename(file.name)}: vacío")
                    continue

                temp_wav = tempfile.mktemp(suffix=".wav")
                temp_mp3 = tempfile.mktemp(suffix=".mp3")

                self.generator.generate(text, temp_wav, verbose=False)
                self.processor.process(temp_wav, temp_mp3, verbose=False)

                output_name = os.path.splitext(os.path.basename(file.name))[0] + ".mp3"
                output_path = os.path.join("output", output_name)
                os.makedirs("output", exist_ok=True)
                shutil.copy(temp_mp3, output_path)

                success += 1
                results.append(f"✓ {output_name}")

                os.remove(temp_wav)
                os.remove(temp_mp3)

            except Exception as e:
                errors += 1
                results.append(f"✗ {os.path.basename(file.name)}: {str(e)[:50]}")

        return f"**Resultados:** {success} OK, {errors} errores\n\n" + "\n".join(results)


def create_ui():
    """Crear interfaz Gradio"""
    studio = AudioStudio()

    with gr.Blocks(
        title="🎙️ VibeVoice Audio Studio",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="green",
        )
    ) as demo:

        gr.Markdown("""
        # 🎙️ VibeVoice Audio Studio
        ## Generador de Audio en Español Colombiano

        Genera audio de alta calidad con vocabulario colombiano auténtico.
        Usa VibeVoice-Realtime con post-procesamiento profesional.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuración")

                load_btn = gr.Button("🔄 Cargar Modelo", variant="primary", size="lg")
                load_status = gr.Textbox(label="Estado", lines=1, interactive=False)

                voice_select = gr.Dropdown(
                    choices=["sp-Spk1_man", "sp-Spk0_woman"],
                    value="sp-Spk1_man",
                    label="🎭 Voz",
                    info="sp-Spk1_man: Masculina, sp-Spk0_woman: Femenina"
                )

                cfg_slider = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.5,
                    step=0.1,
                    label="📊 CFG Scale",
                    info="Mayor = más calidad (y más lento)"
                )

                ddpm_slider = gr.Slider(
                    minimum=3,
                    maximum=10,
                    value=5,
                    step=1,
                    label="🔢 DDPM Steps",
                    info="Mayor = más calidad (y más lento)"
                )

                lufs_slider = gr.Slider(
                    minimum=-20,
                    maximum=-12,
                    value=-16,
                    step=1,
                    label="🔊 LUFS Target",
                    info="Normalización de volumen"
                )

            with gr.Column(scale=2):
                gr.Markdown("### 📝 Generar Audio")

                text_input = gr.Textbox(
                    label="Texto",
                    placeholder="Escriba su texto aquí en español colombiano...\n\nEjemplo: Buenos días mijo, qué tal amaneció. Hoy vamos a hablar de las bondades del ajenjo para la salud.",
                    lines=6,
                )

                with gr.Row():
                    generate_btn = gr.Button("🎙️ Generar Audio", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Limpiar", size="lg")

                status_output = gr.Textbox(label="Estado", lines=2, interactive=False)

                audio_output = gr.Audio(
                    label="🎧 Vista Previa",
                    type="filepath",
                    interactive=False
                )

        gr.Markdown("---")

        gr.Markdown("### 📁 Generación por Lotes")

        with gr.Row():
            batch_files = gr.File(
                file_count="multiple",
                file_types=[".txt"],
                label="Archivos de texto (.txt)"
            )

        batch_btn = gr.Button("📦 Generar Lote", variant="secondary", size="lg")
        batch_output = gr.Textbox(label="Resultados", lines=8, interactive=False)

        gr.Markdown("""
        ---
        ### 📚 Biblioteca de Textos

        Textos de ejemplo en español colombiano:

        **Salud:**
        - `colombian_library/salud/introduccion.txt`
        - `colombian_library/salud/plantas_basicas.txt`
        - `colombian_library/salud/preparaciones.txt`

        **Campo:**
        - `colombian_library/campo/aprendizajes.txt`
        - `colombian_library/campo/sabiduria_rural.txt`

        ---
        ### 💡 Consejos

        - **Texto corto** (<50 palabras): ~15-30 segundos de generación
        - **Texto medio** (50-200 palabras): ~1-3 minutos de generación
        - **Texto largo** (>200 palabras): ~5-10 minutos de generación
        - Sin GPU NVIDIA: El proceso es más lento (~5-10x)
        """)

        def load_model_fn():
            return studio.load_model()

        def generate_fn(text, voice, cfg_scale, ddpm_steps, lufs_target):
            return studio.generate_audio(text, voice, cfg_scale, ddpm_steps, lufs_target)

        def generate_batch_fn(files, voice, cfg_scale, ddpm_steps, lufs_target):
            return studio.generate_batch(files, voice, cfg_scale, ddpm_steps, lufs_target)

        def clear_fn():
            return "", None, ""

        load_btn.click(
            fn=load_model_fn,
            outputs=load_status
        )

        generate_btn.click(
            fn=generate_fn,
            inputs=[text_input, voice_select, cfg_slider, ddpm_slider, lufs_slider],
            outputs=[audio_output, status_output]
        )

        clear_btn.click(
            fn=clear_fn,
            outputs=[text_input, audio_output, status_output]
        )

        batch_btn.click(
            fn=generate_batch_fn,
            inputs=[batch_files, voice_select, cfg_slider, ddpm_slider, lufs_slider],
            outputs=batch_output
        )

    return demo


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VibeVoice Audio Studio")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Crear URL pública")
    parser.add_argument("--debug", action="store_true", help="Modo debug")
    args = parser.parse_args()

    print("=" * 60)
    print("🎙️ VIBEVOICE AUDIO STUDIO")
    print("=" * 60)
    print("Abriendo interfaz gráfica...")
    print()

    demo = create_ui()

    demo.launch(
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()