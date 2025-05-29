import json
import os
import subprocess

from tts_audiobook_tool.ffmpeg_util import FfmpegUtil
from tts_audiobook_tool.util import *

class LoudnessNormalizationUtil:
    """
    "Two-Pass Loudness Normalization (to EBU R128 / ITU-R BS.1770 specifications)"
    """

    DEFAULT_I = -18.0
    DEFAULT_LRA = 7.0
    DEFAULT_TP = -2.0

    @staticmethod
    def normalize(
            source_path: str,
            dest_path: str="",
            i: float=DEFAULT_I,
            lra: float=DEFAULT_LRA,
            tp: float=DEFAULT_TP
    ) -> str:
        """
        Returns error message string on fail, else empty string
        """
        if not dest_path:
            dest_path = source_path # ie, overwrite original


        # Pass 1
        result = LoudnessNormalizationUtil.get_loudness_json(source_path, i, lra, tp)
        if isinstance(result, str):
            return result
        else:
            loudness_stats = result

        # Pass 2
        err = LoudnessNormalizationUtil.do_loudness_transform(source_path, dest_path, loudness_stats)
        return err

    @staticmethod
    def get_loudness_json(
        path: str,
        i: float=DEFAULT_I,
        lra: float=DEFAULT_LRA,
        tp: float=DEFAULT_TP,
        no_params: bool = False
    ) -> dict | str:
        """
        Extracts the FFMPEG loudnorm filter json text output to a dict
        """

        if no_params:
            loudnorm_string = f"loudnorm=print_format=json"
        else:
            loudnorm_string = f"loudnorm=I={i}:LRA={lra}:TP={tp}:print_format=json"

        ffmpeg_cmd = [
            "ffmpeg",
            "-nostats",
            "-hide_banner",
            "-loglevel", "info",
            "-i", path,
            "-af", loudnorm_string,
            "-f", "null",
            "-"
        ]

        # Run ffmpeg and capture stderr (yes, stderr)
        try:
            process = subprocess.Popen(ffmpeg_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            _, stderr = process.communicate()
        except Exception as e:
            return str(e)

        # Simple curly string search
        output = str(stderr)
        start = output.find('{')
        end = output.find('}', start) + 1
        if start != -1 and end != 0:  # Both braces found
            json_str = output[start:end]
        else:
            return "No valid substring found"

        try:
            json_dict = json.loads(json_str)
        except json.JSONDecodeError as e:
            return str(e)

        # Some validation
        for item in ["input_i", "input_tp", "input_lra", "input_thresh", "target_offset"]:
            if not item in json_dict:
                return f"Missing expected field {item}"

        return json_dict

        """
        Example output
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
    def do_loudness_transform(
        input_file_path: str,
        output_file_path: str,
        loudness_stats: dict,
        target_i: float = DEFAULT_I,
        target_lra: float = DEFAULT_LRA,
        target_tp: float = DEFAULT_TP
    ) -> str:
        """
        Performs the second pass of FFmpeg's loudnorm filter to normalize an audio file.
        Assumes mono audio and outputs to FLAC format.

        Args:
            input_file_path (str): Path to the original input audio file.
            output_file_path (str): Path for the normalized output FLAC file.
            loudness_stats (dict): A dictionary containing the loudness statistics
                                from FFmpeg's loudnorm pass 1 (parsed JSON).
                                Expected keys: "input_i", "input_lra", "input_tp",
                                                "input_thresh", "target_offset".
            target_i (float): Target Integrated Loudness in LUFS.
            target_lra (float): Target Loudness Range in LU.
            target_tp (float): Target True Peak in dBTP.

            Rem, i, tp, and tp values must match those used to obtain loudness_stats from "pass 1"

        Returns:
            str: Empty string on success, or an error message string on failure.
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
            "-hide_banner",
            "-loglevel", "error", # Show only errors
            "-i", input_file_path,
            "-af", filter_string,
            "-c:a", "flac",      # Specify FLAC codec
            "-compression_level", "5"
        ]
        err = FfmpegUtil.make_file(partial_command, output_file_path, use_temp_file=True)
        return err

    # ---

    @staticmethod
    def normalize_directory(dir_path: str) -> None:
        """
        Normalizes and overwrites files in a directory
        Prints status
        For development (not used in current UX flow)
        """
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            printt(file_path)
            err = LoudnessNormalizationUtil.normalize(file_path)
            printt(err if err else "ok")

    @staticmethod
    def print_lra_directory(dir_path: str):
        """ For development"""
        count = 0
        sum = 0
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            result = LoudnessNormalizationUtil.get_lra(file_path)
            if isinstance(result, float):
                count += 1
                sum += result
                result = f"{result:.2f}"
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
        return lra