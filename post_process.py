#!/usr/bin/env python3
"""
VibeVoice Post-Processor - Pipeline de Audio Profesional

Pipeline de post-procesamiento para mejorar la calidad del audio generado:
1. Eliminar silencios largos
2. Normalizar a -16 LUFS
3. EQ (claridad de voz)
4. Compresión
5. Fade in/out
6. Exportar a MP3

Uso:
    python post_process.py input.wav output.mp3
    python post_process.py --batch --dir output/
    python post_process.py --config mi_config.json input.wav output.mp3
"""

import os
import sys
import argparse
import json
import glob
import subprocess
from pathlib import Path


DEFAULT_CONFIG = {
    "silence_threshold": -40,
    "silence_duration": 2.0,
    "lufs_target": -16,
    "lufs_true_peak": -1.5,
    "lufs_lra": 11,
    "eq_low_cut": 200,
    "eq_high_boost": 3000,
    "eq_high_boost_db": 2,
    "compression_threshold": -18,
    "compression_ratio": 3,
    "compression_attack": 5,
    "compression_release": 50,
    "fade_in": 100,
    "fade_out": 100,
    "output_format": "mp3",
    "output_bitrate": "192k",
    "output_sample_rate": 24000
}


class AudioPostProcessor:
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG.copy()
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Verificar que FFmpeg esté instalado"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError("FFmpeg no encontrado")
            self.ffmpeg_version = result.stdout.split("ffmpeg version")[1].split("\n")[0] if "ffmpeg version" in result.stdout else "unknown"
            print(f"✓ FFmpeg disponible: {self.ffmpeg_version.strip()}")
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg no está instalado.\n"
                "Instala con: sudo apt install ffmpeg (Linux)\n"
                "O descarga de: https://ffmpeg.org/download.html"
            )

    def process(self, input_path, output_path, verbose=True):
        """Procesar archivo de audio"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Archivo no encontrado: {input_path}")

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"PROCESANDO AUDIO")
            print("=" * 60)
            print(f"Entrada: {input_path}")
            print(f"Salida: {output_path}")

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        temp_files = []

        try:
            temp_wav = output_path + ".temp.wav"
            temp_files.append(temp_wav)

            filters = []

            filters.append(f"silenceremove=stop_periods=-1:stop_threshold={self.config['silence_threshold']}dB:stop_duration={self.config['silence_duration']}")

            filters.append(f"loudnorm=I={self.config['lufs_target']}:TP={self.config['lufs_true_peak']}:LRA={self.config['lufs_lra']}:print_format=json")

            filters.append(
                f"acompressor=threshold={self.config['compression_threshold']}dB:"
                f"ratio={self.config['compression_ratio']}:"
                f"attack={self.config['compression_attack']}:"
                f"release={self.config['compression_release']}"
            )

            if self.config["fade_in"] > 0:
                filters.append(f"afade=t=in:st=0:d={self.config['fade_in']/1000}")

            if self.config["fade_out"] > 0:
                import soundfile as sf
                info = sf.info(input_path)
                duration = info.duration
                fade_start = duration - (self.config['fade_out'] / 1000)
                if fade_start > 0:
                    filters.append(f"afade=t=out:st={fade_start:.3f}:d={self.config['fade_out']/1000}")

            filter_str = ",".join(filters)

            if verbose:
                print(f"\nFiltros aplicados:")
                for f in filters:
                    print(f"  - {f}")

            cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-af", filter_str,
                "-ar", str(self.config["output_sample_rate"]),
                "-ac", "1",
                temp_wav
            ]

            if verbose:
                print(f"\nEjecutando FFmpeg...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print(f"Error FFmpeg: {result.stderr}")
                raise RuntimeError(f"FFmpeg falló: {result.stderr}")

            if verbose:
                for line in result.stderr.split("\n"):
                    if "Parsed_loudnorm" in line or "mean_volume" in line or "max_volume" in line:
                        print(f"  {line.strip()}")

            if self.config["output_format"] == "mp3":
                cmd = [
                    "ffmpeg", "-y",
                    "-i", temp_wav,
                    "-codec:a", "libmp3lame",
                    "-b:a", self.config["output_bitrate"],
                    output_path
                ]
            else:
                import shutil
                shutil.copy(temp_wav, output_path)
                return

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg MP3 falló: {result.stderr}")

            if verbose:
                print(f"\n✓ Audio procesado guardado: {output_path}")

            return output_path

        finally:
            for temp in temp_files:
                if os.path.exists(temp):
                    try:
                        os.remove(temp)
                    except:
                        pass

    def process_batch(self, input_dir, output_dir=None, pattern="*.wav", verbose=True):
        """Procesar múltiples archivos"""
        if output_dir is None:
            output_dir = input_dir

        files = glob.glob(os.path.join(input_dir, pattern))

        if not files:
            if verbose:
                print(f"No se encontraron archivos en {input_dir}")
            return []

        results = []
        total = len(files)

        for i, input_file in enumerate(files):
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            ext = self.config["output_format"]
            output_file = os.path.join(output_dir, f"{base_name}.{ext}")

            if verbose:
                print(f"\n[{i+1}/{total}] {os.path.basename(input_file)}")

            try:
                self.process(input_file, output_file, verbose=verbose)
                results.append({"file": input_file, "output": output_file, "status": "success"})
                if verbose:
                    print(f"  ✓ OK")
            except Exception as e:
                results.append({"file": input_file, "status": "error", "error": str(e)})
                if verbose:
                    print(f"  ✗ ERROR: {e}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="VibeVoice Post-Processor - Audio Profesional",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Archivo único
  python post_process.py input.wav output.mp3

  # Batch
  python post_process.py --batch --dir output/

  # Con configuración custom
  python post_process.py --config mi_config.json input.wav output.mp3
        """
    )

    parser.add_argument("input", nargs="?", help="Archivo de entrada")
    parser.add_argument("output", nargs="?", help="Archivo de salida")
    parser.add_argument("--batch", action="store_true", help="Modo batch")
    parser.add_argument("--dir", type=str, help="Directorio de entrada")
    parser.add_argument("--output_dir", type=str, help="Directorio de salida")
    parser.add_argument("--pattern", type=str, default="*.wav", help="Patrón de archivos")
    parser.add_argument("--config", type=str, help="Archivo de configuración JSON")
    parser.add_argument("--lufs", type=float, help="LUFS target (default: -16)")
    parser.add_argument("--bitrate", type=str, help="Bitrate MP3 (default: 192k)")

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config.update(json.load(f))

    for key in ["lufs", "bitrate"]:
        attr = getattr(args, key, None)
        if attr is not None:
            if key == "lufs":
                config["lufs_target"] = attr
            elif key == "bitrate":
                config["output_bitrate"] = attr

    processor = AudioPostProcessor(config)

    if args.batch and args.dir:
        results = processor.process_batch(args.dir, args.output_dir, args.pattern)

        success = sum(1 for r in results if r["status"] == "success")
        errors = len(results) - success
        print(f"\n{'=' * 60}")
        print(f"RESULTADOS: {success} OK, {errors} errores")
        print("=" * 60)

        for r in results:
            if r["status"] == "error":
                print(f"✗ {r['file']}: {r.get('error', 'Unknown error')}")

    elif args.input and args.output:
        processor.process(args.input, args.output)

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)