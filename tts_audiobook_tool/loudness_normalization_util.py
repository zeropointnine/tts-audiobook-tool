import json
import os
import subprocess

from tts_audiobook_tool.app_metadata import AppMetadata
from tts_audiobook_tool.app_types import NormalizationSpecs
from tts_audiobook_tool.ffmpeg_util import FfmpegUtil
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.util import *

class LoudnessNormalizationUtil:
    """
    Two-Pass Loudness Normalization (to EBU R128 / ITU-R BS.1770 specifications)

    I = integrated loudness - average loudness over whole clip; measured in LUFS
    LRA = loudness range - dynamic range; lower values = less dynamic range / more compression
    TP = true peak - highest peak after processing; acts as a limiter, prevents clipping
    """

    @staticmethod
    def normalize_file(
            source_flac: str,
            specs: NormalizationSpecs,
            dest_path: str=""
    ) -> str:
        """
        Source file must be FLAC.
        Returns error message string on fail, else empty string

        Dest audio codec is chosen based on dest file suffix

        Prints some status
        """
        if not source_flac.lower().endswith(".flac"):
            return "Source file must be .flac"

        if not dest_path:
            dest_path = source_flac # ie, overwrite original

        printt(f"EBU R 128 loudness normalization ({specs.label})")
        printt()

        printt(f"Pass 1, please wait... {COL_DIM}(no feedback shown)")
        printt()
        result = LoudnessNormalizationUtil.get_loudness_json(source_flac, specs.i, specs.lra, specs.tp)
        if isinstance(result, str):
            return result
        else:
            loudness_stats = result

        printt("Pass 2...")
        printt()
        err = LoudnessNormalizationUtil.do_loudness_transform_and_save(
            source_flac, dest_path, loudness_stats, specs.i, specs.lra, specs.tp)
        print()

        return err

    @staticmethod
    def get_loudness_json(
        path: str,
        i: float,
        lra: float,
        tp: float,
        no_params: bool = False
    ) -> dict | str:
        """
        Extracts the FFMPEG loudnorm filter json text output to a dict
        """

        if no_params:
            loudnorm_string = f"loudnorm=print_format=json"
        else:
            loudnorm_string = f"loudnorm=I={i}:LRA={lra}:TP={tp}:print_format=json"

        # Rem, cannot show status here bc corrupts output being captured
        ffmpeg_command = [
            'ffmpeg',
            '-nostats', "-hide_banner", '-i',
            path,
            '-af', loudnorm_string,
            '-f', 'null', '-'
        ]

        # Run ffmpeg and capture stderr (yes, stderr)
        try:
            process = subprocess.run(ffmpeg_command, capture_output=True, text=True, encoding='utf-8')
        except Exception as e:
            return make_error_string(e)
        if process.returncode != 0:
            return f"Ffmpeg error - return code {process.returncode}"

        raw_stderr_string = process.stderr
        try:
            start_index = raw_stderr_string.rfind('{')
            end_index = raw_stderr_string.rfind('}') + 1
            clean_json_string = raw_stderr_string[start_index:end_index]
            loudness_data = json.loads(clean_json_string)
        except Exception as e:
            return f"Error: {e}\nRaw output:\n{raw_stderr_string}"

        # Some validation
        for item in ["input_i", "input_tp", "input_lra", "input_thresh", "target_offset"]:
            if not item in loudness_data:
                return f"Missing expected field {item}"

        return loudness_data

        """
        Example output:

            {
                "input_i": "-18.44",
                "input_tp": "-1.08",
                "input_lra": "1.20",
                "input_thresh": "-28.65",
                "output_i": "-18.04",
                "output_tp": "-2.00",
                "output_lra": "0.00",
                "output_thresh": "-28.30",
                "normalization_type": "dynamic",
                "target_offset": "0.04"
            }

        input_i - measured integrated loudness
        input_tp - measured true peak
        input_lra - measured loudness range
        input_thresh - loudness threshold or smth
        target_offset - "This is the calculated gain offset in dB that loudnorm has determined should be applied to the file to reach the target integrated loudness (I), while also considering the LRA and TP constraints."
        """

    @staticmethod
    def do_loudness_transform_and_save(
        input_flac_path: str,
        output_file_path: str,
        loudness_stats: dict,
        target_i: float,
        target_lra: float,
        target_tp: float
    ) -> str:
        """
        Returns error message on failure else empty string.

        Performs the second pass of FFmpeg's loudnorm filter to normalize an audio file.
        Assumes mono audio.
        Source file must be FLAC.
        Exports to FLAC or AAC based on output file path suffix.

        Args:
            loudness_stats (dict): A dictionary containing the loudness statistics
                                from FFmpeg's loudnorm pass 1 (parsed JSON).
                                Expected keys: "input_i", "input_lra", "input_tp",
                                                "input_thresh", "target_offset".
            target_i (float): Target Integrated Loudness in LUFS.
            target_lra (float): Target Loudness Range in LU.
            target_tp (float): Target True Peak in dBTP.

            Rem, i, lra, and tp values must match those used to obtain loudness_stats from "pass 1"
        """

        if not os.path.exists(input_flac_path):
            return f"File not found: {input_flac_path}"

        required_keys = ["input_i", "input_lra", "input_tp", "input_thresh", "target_offset"]
        for key in required_keys:
            if key not in loudness_stats:
                return f"Error: Missing key '{key}' in loudness_stats dictionary."

        # Extract measured values from the stats dictionary
        # FFmpeg expects these as strings in the filter, which they already are from JSON
        measured_i = loudness_stats["input_i"]
        measured_lra = loudness_stats["input_lra"]
        measured_tp = loudness_stats["input_tp"]
        measured_thresh = loudness_stats["input_thresh"]
        offset = loudness_stats["target_offset"] # This is the 'target_offset' from JSON

        # Construct the -af loudnorm filter string
        # Note: FFmpeg filter options are typically strings, so the values
        # from loudness_stats (which are strings from JSON) are fine.
        # The target values are floats, but f-string formatting will convert them.
        filter_string = (
            f"loudnorm="
            f"I={target_i}:"
            f"LRA={target_lra}:"
            f"TP={target_tp}:"
            f"measured_I={measured_i}:"
            f"measured_LRA={measured_lra}:"
            f"measured_TP={measured_tp}:"
            f"measured_thresh={measured_thresh}:"
            f"offset={offset}"
        )

        partial_command = [
            "-y",  # Overwrite output file if it exists
            "-hide_banner", "-loglevel", "error", "-stats",
            "-i", input_flac_path,
            "-af", filter_string
        ]

        output_suffix = Path(output_file_path).suffix.lower()
        if output_suffix == ".flac":
            partial_command.extend(FFMPEG_ARGUMENTS_OUTPUT_FLAC)
        elif output_suffix in AAC_SUFFIXES:
            partial_command.extend(FFMPEG_ARGUMENTS_OUTPUT_AAC)
        else:
            return "Unsupported output type"

        err = FfmpegUtil.make_file(partial_command, output_file_path, use_temp_file=True)
        if err:
            return err

        if output_suffix in AAC_SUFFIXES:
            # Re-apply app metadata from source flac file
            # (Only required for AAC; when dest is flac, the metadata does get passed along and preserved properly)
            meta = AppMetadata.load_from_flac(input_flac_path)
            if meta:
                err = AppMetadata.save_to_mp4(meta, output_file_path)
                if err:
                    return err

        return "" # success
