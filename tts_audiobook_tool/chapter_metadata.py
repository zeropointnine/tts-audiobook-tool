from __future__ import annotations

from tts_audiobook_tool.app_types import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.util import *

class ChapterMetadata:
    """
    Creates M4B chapter metadata
    """

    @staticmethod
    def make_metadata(
            project: Project, 
            durations: list[float],
            file_title: str,
            artist: str=APP_NAME
    ) -> str:
        """
        Creates M4B chapter metadata string
        """
        
        sum = 0
        duration_sums = []
        for i in range(len(durations)):
            duration_sums.append(sum)
            sum += durations[i]
            if i == len(durations) - 1:
                duration_sums.append(sum) # Add end item

        indices = list(project.section_dividers)
        indices.insert(0, 0) # insert index 0 at the beginning

        chapter_info = []
        for i, index in enumerate(indices):
            start_time = duration_sums[index]
            if i == len(indices) - 1:
                end_time = duration_sums[-1]
            else:
                end_time = duration_sums[index + 1]
            chapter_title = project.phrase_groups[index].presentable_text
            chapter_title = ellipsize(chapter_title, 60) # TODO: Consider prefixing with [9999] etc
            info = (start_time, end_time, chapter_title)
            chapter_info.append(info)

        meta = ";FFMETADATA1\n"
        meta += f"title={file_title}\n"
        meta += f"artist={artist}\n\n"

        for start_time, end_time, chapter_title in chapter_info:
            section  = "[CHAPTER]\n"
            section += "TIMEBASE=1/1000\n"
            section += f"START={int(start_time * 1000)}\n"
            section += f"END={int(end_time * 1000)}\n"
            section += f"title={chapter_title}\n\n"
            meta += section
                
        return meta
    
    @staticmethod
    def make_copy_with_metadata(source_path: str, dest_path: str, metadata: str) -> str:
        """
        Makes a copy of an M4A/M4B/MP4 file with added M4B chapter metadata.
        Using ffmpeg is the most compatible approach for doing this.
        Returns error string if any.
        """
        if not os.path.exists(source_path):
            return f"Source file does not exist: {source_path}"
        
        if not source_path.lower().endswith(tuple(AAC_SUFFIXES)):
            return f"Source file has incorrect filename suffix: {source_path}"
        
        full_command = [
            FFMPEG_COMMAND,
            "-y",                 # Overwrite 
            "-i", source_path,    # Input 0: The original file
            "-i", "-",            # Input 1: Standard input (the 'meta' string)
            "-map", "0",          # Map all streams from source
            "-map_metadata", "0", # Keep the original global metadata
            "-map_chapters", "1", # Take the new chapters from the pipe
            "-c", "copy",         # Copy streams without re-encoding
            "-movflags", "use_metadata_tags", # Preserve custom tags
            dest_path        
        ]

        try:
            # Use 'input' to pass the string to stdin. 
            # 'capture_output' ensures e.stderr is populated on failure.
            subprocess.run(
                full_command,
                input=metadata,
                check=True,
                text=True,
                encoding='utf-8',
                capture_output=True
            )

        except subprocess.CalledProcessError as e:
            err_detail = e.stderr.splitlines()[-1] if e.stderr else "Unknown error"
            return f"Ffmpeg error {e.returncode}: {err_detail}"

        except Exception as e:
            return f"Subprocess error: {str(e)}"
        
        return ""
