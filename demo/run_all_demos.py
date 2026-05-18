#!/usr/bin/env python3
"""
Script maestro para ejecutar todas las demos de VibeVoice

Uso:
    python demo/run_all_demos.py          # Ejecutar todas las demos
    python demo/run_all_demos.py --check   # Solo verificar requisitos
    python demo/run_all_demos.py --asr      # Solo demo ASR
    python demo/run_all_demos.py --gradio  # Solo demo Gradio
    python demo/run_all_demos.py --tts      # Solo demo TTS
"""

import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_command(cmd, description):
    print("\n" + "=" * 60)
    print(f"🚀 {description}")
    print("=" * 60)
    print(f"Comando: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def check_requirements():
    print("\n" + "=" * 60)
    print("🔍 VERIFICANDO REQUISITOS DEL SISTEMA")
    print("=" * 60)

    check_script = os.path.join(os.path.dirname(__file__), "check_requirements.py")
    result = subprocess.run([sys.executable, check_script])
    return result.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VibeVoice Demo Runner")
    parser.add_argument("--check", action="store_true", help="Solo verificar requisitos")
    parser.add_argument("--asr", action="store_true", help="Ejecutar solo demo ASR")
    parser.add_argument("--gradio", action="store_true", help="Ejecutar solo demo Gradio")
    parser.add_argument("--tts", action="store_true", help="Ejecutar solo demo TTS")
    parser.add_argument("--skip-download", action="store_true", help="Omitir descarga de modelos")
    args = parser.parse_args()

    demo_dir = os.path.dirname(__file__)

    if args.check:
        check_requirements()
        return

    if not args.skip_download:
        print("\n" + "=" * 60)
        print("NOTA: La primera ejecución descargará modelos de HuggingFace")
        print("Esto puede tomar varios minutos dependiendo de tu conexión.")
        print("=" * 60)

    all_demos = not (args.asr or args.gradio or args.tts)

    if args.asr or all_demos:
        run_command(
            f'cd {demo_dir} && python run_asr_demo.py --device cpu',
            "DEMO 1: ASR - Transcripción de Audio"
        )

    if args.gradio or all_demos:
        print("\n⚠️  Para ejecutar el demo Gradio, usa:")
        print(f"   python {demo_dir}/run_gradio_demo.py")
        print("   Luego abre http://localhost:7860 en tu navegador")

    if args.tts or all_demos:
        run_command(
            f'cd {demo_dir} && python run_colombian_tts.py --device cpu',
            "DEMO 3: TTS - Síntesis de Voz (Español Colombiano)"
        )

    print("\n" + "=" * 60)
    print("📋 RESUMEN DE DEMOS")
    print("=" * 60)
    print("""
Demos disponibles:

1. ASR (Transcripción de Audio)
   Script: demo/run_asr_demo.py
   Función: Transcribe archivos de audio hasta 60 minutos

2. Gradio (Interfaz Web Interactiva)
   Script: demo/run_gradio_demo.py
   Función: Interfaz web para probar transcripción

3. TTS Realtime (Síntesis de Voz - Español Colombiano)
   Script: demo/run_colombian_tts.py
   Función: Genera audio desde texto en español colombiano

Comandos rápidos:
   python demo/check_requirements.py   # Verificar sistema
   python demo/run_asr_demo.py           # Demo ASR
   python demo/run_gradio_demo.py --share  # Demo web
   python demo/run_colombian_tts.py        # Demo TTS colombiano
""")


if __name__ == "__main__":
    main()