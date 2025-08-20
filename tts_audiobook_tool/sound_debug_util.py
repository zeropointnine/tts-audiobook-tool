import numpy as np
from PIL import Image, ImageDraw

from tts_audiobook_tool.app_types import Sound, Word

class SoundDebugUtil:

    @staticmethod
    def save_word_timestamps_visualization(sound_data: np.ndarray, sr: int, words: list[Word], dest_path_png: str):
        """
        Creates and saves waveform visualization image as a PNG file.
        For each timestamp pair, a green and red vertical line should be drawn.
        The index number and the corresponding "word" are drawn next to each vertical line.
        """
        horizontal_padding = 10

        # Calculate image dimensions
        duration_seconds = len(sound_data) / sr
        waveform_width = int(duration_seconds * 500)
        image_width = waveform_width + (2 * horizontal_padding)
        image_height = 300

        # Create a blank image
        image = Image.new('RGB', (image_width, image_height), 'white')
        draw = ImageDraw.Draw(image)

        # Normalize the sound data to fit the image height
        max_amplitude = np.max(np.abs(sound_data))
        if max_amplitude == 0:
            max_amplitude = 1.0  # Avoid division by zero for silent audio
        normalized_data = sound_data / max_amplitude

        # Draw the waveform
        for i in range(1, len(normalized_data)):
            x1 = horizontal_padding + int((i - 1) / len(normalized_data) * waveform_width)
            y1 = int((1 - normalized_data[i - 1]) / 2 * image_height)
            x2 = horizontal_padding + int(i / len(normalized_data) * waveform_width)
            y2 = int((1 - normalized_data[i]) / 2 * image_height)
            draw.line((x1, y1, x2, y2), fill='black', width=1)

        # Draw the vertical lines, index numbers, and words for timestamps
        for index, word_timing in enumerate(words):
            start = word_timing.start
            end = word_timing.end
            word = word_timing.word

            start_x = horizontal_padding + int(start * 500)
            end_x = horizontal_padding + int(end * 500)
            index_text = str(index)

            # Green line: 100px tall from the top
            draw.line((start_x, 0, start_x, 100), fill='green', width=2)
            # Draw the index number next to the green line
            draw.text((start_x + 5, 5), index_text, fill='green')
            # Draw the word below the index
            draw.text((start_x + 5, 20), word, fill='green')

            # Red line: 100px tall from the bottom
            draw.line((end_x, image_height - 100, end_x, image_height), fill='red', width=2)
            # Draw the index number above the red line
            draw.text((end_x + 5, image_height - 120), index_text, fill='red')

        # Save the image
        image.save(dest_path_png)

        if False:
            # Print input data to console, too
            for i, word in enumerate(words):
                print(f"{i} {word.word} {word.start:.2f}-{word.end:.2f}")


    @staticmethod
    def save_local_minima_visualization(sound: Sound, target_timestamp: float, local_minima: float, dest_path_png: str):
        """
        Creates waveform image that is +/-500ms from target_timestamp
        Red line is target_timestamp. Green line is local_minima.
        For use while debugging, etc.
        """

        # Image parameters
        width, height = 1200, 600
        background_color = (255, 255, 255)
        waveform_color = (0, 0, 255)
        target_line_color = (255, 0, 0)
        minima_line_color = (0, 255, 0)
        center_line_color = (200, 200, 200)

        # Create a new image
        img = Image.new('RGB', (width, height), background_color)
        draw = ImageDraw.Draw(img)

        # Define the window to display
        display_window_ms = 500
        display_window_s = display_window_ms / 1000

        # Calculate sample indices for the display window
        start_s = target_timestamp - display_window_s
        end_s = target_timestamp + display_window_s
        start_sample = int(start_s * sound.sr)
        end_sample = int(end_s * sound.sr)

        # Clamp to audio boundaries
        start_sample = max(0, start_sample)
        end_sample = min(len(sound.data), end_sample)

        # Extract the waveform segment
        waveform_segment = sound.data[start_sample:end_sample]

        # --- Drawing ---

        # Draw center line
        center_y = height // 2
        draw.line([(0, center_y), (width, center_y)], fill=center_line_color, width=1)

        # Draw waveform
        num_samples = len(waveform_segment)
        if num_samples > 1:
            # Downsample for performance if necessary, or draw all points
            # For simplicity, we'll map each sample to a horizontal position
            for i in range(num_samples - 1):
                x1 = int(i / num_samples * width)
                y1 = center_y - int(waveform_segment[i] * center_y * 0.9) # 0.9 to avoid touching edges

                x2 = int((i + 1) / num_samples * width)
                y2 = center_y - int(waveform_segment[i+1] * center_y * 0.9)

                draw.line([(x1, y1), (x2, y2)], fill=waveform_color, width=2)

        # Draw target timestamp indicator
        target_pos_s = target_timestamp - start_s
        target_x = int((target_pos_s / (end_s - start_s)) * width)
        draw.line([(target_x, 0), (target_x, height)], fill=target_line_color, width=2)

        # Draw local minima indicator
        minima_pos_s = local_minima - start_s
        minima_x = int((minima_pos_s / (end_s - start_s)) * width)
        draw.line([(minima_x, 0), (minima_x, height)], fill=minima_line_color, width=3)

        # Save
        img.save(dest_path_png)
        print(f"Waveform visualization saved to: {dest_path_png}")
