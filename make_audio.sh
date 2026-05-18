#!/bin/bash
#
# VibeVoice make_audio.sh - Workflow Completo de Audio para Videos
#
# Combina generación TTS + post-procesamiento en un solo comando.
#
# Uso:
#   ./make_audio.sh "Mi texto aquí" output.mp3
#   ./make_audio.sh --text "Texto" --output audio.mp3
#   ./make_audio.sh --file texto.txt --output audio/
#   ./make_audio.sh --interactive
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATE_TTS="python $SCRIPT_DIR/generate_tts.py"
POST_PROCESS="python $SCRIPT_DIR/post_process.py"

show_help() {
    cat << EOF
VibeVoice Audio Workflow - Audio de Alta Calidad para Videos

Uso:
    $0 "Mi texto aquí" output.mp3          # Texto directo
    $0 --text "Texto" --output audio.mp3   # Con flags
    $0 --file texto.txt --output audio/    # Desde archivo
    $0 --batch --dir textos/ --output out/ # Batch mode
    $0 --interactive                       # Modo interactivo

Opciones:
    --text TEXT         Texto a sintetizar
    --file FILE         Archivo de texto de entrada
    --batch             Modo batch (directorio)
    --dir DIRECTORIO     Directorio con archivos .txt
    --output OUTPUT     Archivo o directorio de salida
    --config FILE        Archivo de configuración JSON
    --voice VOICE        Voz (sp-Spk0_woman, sp-Spk1_man)
    --cfg-scale SCALE    CFG scale (default: 1.5)
    --ddpm-steps STEPS   DDPM steps (default: 5)
    --lufs LUFS          LUFS target (default: -16)
    --bitrate BITRATE    Bitrate MP3 (default: 192k)
    --keep-wav           Mantener archivo WAV intermedio
    --help               Mostrar esta ayuda

Voces disponibles:
    sp-Spk0_woman  - Femenina española
    sp-Spk1_man    - Masculina española (default)

Ejemplos:
    # Texto directo
    $0 "Buenos días mijo, qué tal amaneció" audio.mp3

    # Desde archivo
    $0 --file texto.txt --output audio/

    # Batch completo
    $0 --batch --dir textos_colombianos/ --output audios_procesados/

    # Con configuración custom
    $0 --text "Hola" --output hola.mp3 --cfg-scale 1.6 --lufs -14

EOF
}

log() {
    echo -e "\033[0;32m✓\033[0m $1"
}

error() {
    echo -e "\033[0;31m✗ ERROR:\033[0m $1" >&2
}

TEMP_WAV=""
OUTPUT_FILE=""

cleanup() {
    if [ -n "$TEMP_WAV" ] && [ -f "$TEMP_WAV" ] && [ "$KEEP_WAV" != "1" ]; then
        rm -f "$TEMP_WAV" 2>/dev/null || true
    fi
}

trap cleanup EXIT

KEEP_WAV=""
BATCH_MODE=""
INPUT_TEXT=""
INPUT_FILE=""
OUTPUT=""
CONFIG_FILE=""
VOICE="sp-Spk1_man"
CFG_SCALE="1.5"
DDPM_STEPS="5"
LUFS="-16"
BITRATE="192k"

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --text)
            INPUT_TEXT="$2"
            shift 2
            ;;
        --file)
            INPUT_FILE="$2"
            shift 2
            ;;
        --batch)
            BATCH_MODE="1"
            shift
            ;;
        --dir)
            BATCH_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --voice)
            VOICE="$2"
            shift 2
            ;;
        --cfg-scale)
            CFG_SCALE="$2"
            shift 2
            ;;
        --ddpm-steps)
            DDPM_STEPS="$2"
            shift 2
            ;;
        --lufs)
            LUFS="$2"
            shift 2
            ;;
        --bitrate)
            BITRATE="$2"
            shift 2
            ;;
        --keep-wav)
            KEEP_WAV="1"
            shift
            ;;
        --interactive)
            echo "=== VibeVoice Audio Workflow ==="
            echo -n "Ingrese el texto a sintetizar: "
            read INPUT_TEXT
            echo -n "Nombre del archivo de salida: "
            read OUTPUT
            ;;
        -*)
            error "Opción desconocida: $1"
            show_help
            exit 1
            ;;
        *)
            if [ -z "$INPUT_TEXT" ]; then
                INPUT_TEXT="$1"
            elif [ -z "$OUTPUT" ]; then
                OUTPUT="$1"
            fi
            shift
            ;;
    esac
done

if [ "$BATCH_MODE" = "1" ]; then
    if [ -z "$BATCH_DIR" ]; then
        error "Falta --dir para modo batch"
        exit 1
    fi

    OUTPUT_DIR="${OUTPUT:-.}"
    mkdir -p "$OUTPUT_DIR"

    log "Buscando archivos .txt en $BATCH_DIR..."

    files=($(find "$BATCH_DIR" -maxdepth 1 -name "*.txt" -type f 2>/dev/null || true))

    if [ ${#files[@]} -eq 0 ]; then
        error "No se encontraron archivos .txt en $BATCH_DIR"
        exit 1
    fi

    log "Encontrados ${#files[@]} archivos"

    success=0
    errors=0

    for file in "${files[@]}"; do
        basename=$(basename "$file" .txt)
        wav_file="$OUTPUT_DIR/${basename}.wav"
        mp3_file="$OUTPUT_DIR/${basename}.mp3"

        echo ""
        echo "=== Procesando: $basename ==="

        $GENERATE_TTS \
            --file "$file" \
            --output "$wav_file" \
            --voice "$VOICE" \
            --cfg_scale "$CFG_SCALE" \
            --ddpm_steps "$DDPM_STEPS" \
            2>/dev/null

        if [ -f "$wav_file" ]; then
            $POST_PROCESS \
                "$wav_file" "$mp3_file" \
                --lufs "$LUFS" \
                --bitrate "$BITRATE" \
                2>/dev/null

            if [ -f "$mp3_file" ]; then
                log "Generado: $mp3_file"
                ((success++)) || true

                if [ "$KEEP_WAV" != "1" ]; then
                    rm -f "$wav_file"
                fi
            else
                error "Post-process falló para $basename"
                ((errors++)) || true
            fi
        else
            error "Generación falló para $basename"
            ((errors++)) || true
        fi
    done

    echo ""
    echo "========================================="
    echo "RESULTADOS: $success OK, $errors errores"
    echo "========================================="

    exit 0
fi

if [ -n "$INPUT_FILE" ]; then
    if [ ! -f "$INPUT_FILE" ]; then
        error "Archivo no encontrado: $INPUT_FILE"
        exit 1
    fi

    if [ -z "$OUTPUT" ]; then
        basename=$(basename "$INPUT_FILE" .txt)
        OUTPUT="$basename.mp3"
    fi

    TEMP_WAV="${OUTPUT%.mp3}.wav"

    log "Generando audio desde archivo..."
    $GENERATE_TTS \
        --file "$INPUT_FILE" \
        --output "$TEMP_WAV" \
        --voice "$VOICE" \
        --cfg_scale "$CFG_SCALE" \
        --ddpm_steps "$DDPM_STEPS"

elif [ -n "$INPUT_TEXT" ]; then
    if [ -z "$OUTPUT" ]; then
        error "Falta --output"
        exit 1
    fi

    TEMP_WAV="${OUTPUT%.mp3}.wav"

    log "Generando audio desde texto..."
    $GENERATE_TTS \
        --text "$INPUT_TEXT" \
        --output "$TEMP_WAV" \
        --voice "$VOICE" \
        --cfg_scale "$CFG_SCALE" \
        --ddpm_steps "$DDPM_STEPS"

else
    error "Falta texto o archivo de entrada"
    show_help
    exit 1
fi

if [ -f "$TEMP_WAV" ]; then
    log "Post-procesando audio..."
    $POST_PROCESS "$TEMP_WAV" "$OUTPUT" --lufs "$LUFS" --bitrate "$BITRATE"

    if [ -f "$OUTPUT" ]; then
        log "Audio final guardado: $OUTPUT"

        if [ "$KEEP_WAV" != "1" ]; then
            rm -f "$TEMP_WAV"
        fi

        ls -lh "$OUTPUT" 2>/dev/null || true
    else
        error "Post-process falló"
        exit 1
    fi
else
    error "Generación falló"
    exit 1
fi

log "¡Listo!"