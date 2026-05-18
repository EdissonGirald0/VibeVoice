#!/usr/bin/env python3
"""
VibeVoice - Configuración de Voces Personalizadas

Este módulo documenta cómo crear voces personalizadas para VibeVoice.
Actualmente el sistema soporta voces pre-entrenadas.

Voces disponibles en español:
    - sp-Spk0_woman: Voz femenina española
    - sp-Spk1_man: Voz masculina española

Para crear una voz colombiana auténtica, se requiere fine-tuning.
"""

SPANISH_VOICES = {
    "sp-Spk0_woman": {
        "description": "Voz femenina en español",
        "gender": "female",
        "language": "spanish",
        "file": "sp-Spk0_woman.pt"
    },
    "sp-Spk1_man": {
        "description": "Voz masculina en español",
        "gender": "male",
        "language": "spanish",
        "file": "sp-Spk1_man.pt"
    },
}

COLOMBIAN_VOICE_CONFIGS = {
    "colombian_campesino": {
        "base_voice": "sp-Spk1_man",
        "description": "Voz masculina colombiana campesina",
        "characteristics": {
            "pace": "moderate",
            "pitch": "natural",
            "tone": "warm"
        }
    },
    "colombian_woman": {
        "base_voice": "sp-Spk0_woman",
        "description": "Voz femenina colombiana",
        "characteristics": {
            "pace": "moderate",
            "pitch": "natural",
            "tone": "warm"
        }
    }
}

def get_voice_config(name: str = "colombian_campesino"):
    return COLOMBIAN_VOICE_CONFIGS.get(name, COLOMBIAN_VOICE_CONFIGS["colombian_campesino"])


VOICE_RECOMMENDATIONS = """
===========================================
RECOMENDACIONES PARA VOZ COLOMBIANA
===========================================

OPCIONES ACTUALES (sin fine-tuning):
------------------------------------
1. sp-Spk1_man: Mejor para texto masculino/campesino
2. sp-Spk0_woman: Mejor para texto femenino

El modelo base está entrenado en español de España.
Para español colombiano auténtico, se recomienda:

MÉTODO 1: FINE-TUNING (Recomendado)
-----------------------------------
1. Recopilar 30-60 minutos de audio colombiano
2. Preprocesar con herramienta de transcripción
3. Ejecutar fine-tuning:

   torchrun --nproc_per_node=1 finetuning-asr/lora_finetune.py \\
       --model_path microsoft/VibeVoice-Realtime-0.5B \\
       --data_dir ./colombian_voice_data \\
       --output_dir ./colombian_voice_output

MÉTODO 2: SPEAKER CLONING
-------------------------
Si tienes una voz colombiana de referencia:
1. Graba 10-20 segundos de audio de referencia
2. Usa servicios de speaker cloning
3. Genera nuevo archivo .pt de embeddings

MÉTODO 3: PROMPT ENGINEERING
----------------------------
Usar texto que guíe el tono deseado:
- Usar diminutivos: "mijo", "mija", "parcero"
- Expresiones colombianas: "qué vaina", "bardissimo"
- Ritmo pausado y amigable
"""


if __name__ == "__main__":
    print(VOICE_RECOMMENDATIONS)

    print("\nVoces en español disponibles:")
    for key, voice in SPANISH_VOICES.items():
        print(f"  {key}: {voice['description']} ({voice['gender']})")

    print("\nConfigs de voz colombiana:")
    for key, cfg in COLOMBIAN_VOICE_CONFIGS.items():
        print(f"  {key}: {cfg['description']}")
        print(f"    Voz base: {cfg['base_voice']}")