import json
import os
import subprocess

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

    # Tracks with 'ACX standard' somewhat
    # TARGET_I = -19.0
    # TARGET_LRA = 9.0
    # TARGET_TP = -3.0


    # A little more 'aggressive' than 'ACX standard':
    TARGET_I = -17.0
    TARGET_LRA = 7.0
    TARGET_TP = -2.5

    @staticmethod
    def normalize_file(
            source_path: str,
            dest_path: str="",
            i: float=TARGET_I,
            lra: float=TARGET_LRA,
            tp: float=TARGET_TP
    ) -> str:
        """
        Source file must be FLAC.
        Returns error message string on fail, else empty string

        Prints some status
        """
        if not source_path.lower().endswith(".flac"):
            return "Source file must be .flac"

        if not dest_path:
            dest_path = source_path # ie, overwrite original

        printt("EBU R128 normalization pass 1, please wait (no feedback shown)...")

        result = LoudnessNormalizationUtil.get_loudness_json(source_path, i, lra, tp)
        if isinstance(result, str):
            return result
        else:
            loudness_stats = result

        printt("EBU R128 normalization pass 2...")
        printt()

        err = LoudnessNormalizationUtil.do_loudness_transform_and_save(source_path, dest_path, loudness_stats)
        print()

        return err

    @staticmethod
    def get_loudness_json(
        path: str,
        i: float=TARGET_I,
        lra: float=TARGET_LRA,
        tp: float=TARGET_TP,
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
            return str(e)
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
        input_file_path: str,
        output_file_path: str,
        loudness_stats: dict,
        target_i: float = TARGET_I,
        target_lra: float = TARGET_LRA,
        target_tp: float = TARGET_TP
    ) -> str:
        """
        Returns error message on failure

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

        if not os.path.exists(input_file_path):
            return f"File not found: {input_file_path}"

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
            "-i", input_file_path,
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
        return err

    # ---

    @staticmethod
    def print_lra_directory(dir_path: str):
        """ For development """
        count = 0
        sum = 0
        for file in os.listdir(dir_path):
            if not file.endswith(".flac"):
                continue
            file_path = os.path.join(dir_path, file)

            result = LoudnessNormalizationUtil.get_lra(file_path)
            if isinstance(result, float):
                count += 1
                sum += result
                result = f"{result:.2f}"
            else:
                print("error?", result)
            print(result, file_path)

        print("\navg:", (sum / count))


    @staticmethod
    def get_lra(file_path: str) -> float | str:
        """
        Returns LRA or error string
        LRA = EBU R128 Loudness Range
        FYI, this is the value shown in Roon for albums and tracks ("Dynamic range (R128)")
        """
        result = LoudnessNormalizationUtil.get_loudness_json(file_path, no_params=True)
        if isinstance(result, str):
            return result
        lra = result["input_lra"]
        try:
            lra = float(lra)
        except:
            return f"Parse error on {lra}"
        return lra
