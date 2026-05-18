#!/usr/bin/env python3
"""
Verificador de requisitos para ejecutar demos de VibeVoice
"""

import sys
import subprocess
import os

def check_python():
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ requerido")
        return False
    print("✓ Python version OK")
    return True

def check_pytorch():
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA disponible: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠ CUDA no disponible (usará CPU - será lento)")
            return True
    except ImportError:
        print("❌ PyTorch no instalado")
        return False

def check_transformers():
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
        if transformers.__version__ >= "4.51.3":
            print("✓ Transformers version OK")
            return True
        else:
            print(f"❌ Transformers {transformers.__version__} - se requiere >=4.51.3")
            return False
    except ImportError:
        print("❌ Transformers no instalado")
        return False

def check_gpu_memory():
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_memory_gb = props.total_memory / (1024**3)
            print(f"GPU Memory: {total_memory_gb:.1f} GB")
            if total_memory_gb < 16:
                print("⚠ Se recomienda 16GB+ para modelos de 7B")
                return True
            print("✓ GPU memory OK")
            return True
    except:
        pass
    return True

def check_dependencies():
    required = [
        "librosa",
        "pydub",
        "scipy",
        "gradio",
        "fastapi",
        "uvicorn",
        "av"
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            print(f"❌ {pkg} (falta)")
            missing.append(pkg)
    return len(missing) == 0, missing

def check_audio_files():
    demo_dir = os.path.join(os.path.dirname(__file__), "asr_demo")
    if not os.path.exists(demo_dir):
        print("❌ Directorio demo/asr_demo no encontrado")
        return False

    audio_files = [f for f in os.listdir(demo_dir) if f.endswith(('.mp3', '.mp4', '.wav'))]
    if not audio_files:
        print("❌ No se encontraron archivos de audio en demo/asr_demo/")
        return False

    print(f"✓ {len(audio_files)} archivos de audio encontrados:")
    for f in audio_files:
        path = os.path.join(demo_dir, f)
        size_mb = os.path.getsize(path) / (1024*1024)
        print(f"  - {f} ({size_mb:.1f} MB)")
    return True

def main():
    print("=" * 60)
    print("VIBEVOICE - Verificación de Requisitos")
    print("=" * 60)
    print()

    checks = [
        ("Python", check_python()),
        ("PyTorch + CUDA", check_pytorch()),
        ("Transformers", check_transformers()),
        ("GPU Memory", check_gpu_memory()),
    ]

    deps_ok, missing = check_dependencies()
    checks.append(("Dependencias", deps_ok))

    audio_ok = check_audio_files()
    checks.append(("Archivos de Audio", audio_ok))

    print()
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)

    all_passed = True
    for name, passed in checks:
        status = "✓" if passed else "❌"
        print(f"  {status} {name}")

    if missing:
        print()
        print("Dependencias faltantes. Instalar con:")
        print(f"  pip install {' '.join(missing)}")

    print()
    if all(passed for _, passed in checks):
        print("✓ Sistema listo para ejecutar demos")
        return 0
    else:
        print("⚠ Sistema no está completamente configurado")
        print("  Las demos pueden ejecutarse con limitaciones")
        return 1

if __name__ == "__main__":
    sys.exit(main())