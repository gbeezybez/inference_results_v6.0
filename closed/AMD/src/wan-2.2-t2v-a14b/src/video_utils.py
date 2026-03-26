import tempfile
import os
import imageio
from typing import Union, List, Optional
import numpy as np
import PIL.Image

def export_to_video_bytes(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]],
    fps: int = 10,
    quality: float = 5.0,
    bitrate: Optional[int] = None,
    macro_block_size: Optional[int] = 16,
) -> bytes:
    """
    Encode video frames to MP4 bytes using a temporary file.
    
    This approach works reliably across all imageio versions and has
    negligible overhead on modern systems.
    
    Args:
        video_frames: List of video frames (either numpy arrays or PIL Images)
        fps: Frames per second
        quality: Video quality (0-10, higher is better). Lower = faster encoding
        bitrate: Optional fixed bitrate (overrides quality if set)
        macro_block_size: Size constraint for video dimensions (default: 16)
    
    Returns:
        bytes: MP4 encoded video as bytes
    """   
    
    try:
        imageio.plugins.ffmpeg.get_exe()
    except AttributeError:
        raise AttributeError(
            "Unable to find a compatible ffmpeg installation. Please install via `pip install imageio-ffmpeg`"
        )
    
    # Convert frames to uint8 numpy arrays
    if isinstance(video_frames[0], np.ndarray):
        # Assuming frames are normalized [0, 1], scale to [0, 255]
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]
    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    
    # Use temporary file for encoding
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Write video to temporary file
        with imageio.get_writer(
            tmp_path,
            fps=fps,
            quality=quality,
            bitrate=bitrate,
            macro_block_size=macro_block_size
        ) as writer:
            for frame in video_frames:
                writer.append_data(frame)
        
        # Read the file back as bytes
        with open(tmp_path, 'rb') as f:
            mp4_bytes = f.read()
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass  # Ignore cleanup errors
    
    return mp4_bytes