# ComfyUI VIDEO Type Definition and Structure

## Overview
The VIDEO type is ComfyUI's native type for video data. It is a composite type that encapsulates video frames, frame rate, and optional audio and metadata components.

## VideoComponents Structure (Core Definition)

From `comfy_api/latest/_util/video_types.py`:

```python
@dataclass
class VideoComponents:
    """
    Dataclass representing the components of a video.
    """
    images: ImageInput              # Image tensor [N, H, W, 3] in [0, 1] float
    frame_rate: Fraction            # Frames per second (e.g., Fraction(30, 1))
    audio: Optional[AudioInput] = None     # Optional audio dict
    metadata: Optional[dict] = None        # Optional metadata dict
    alpha: Optional[MaskInput] = None      # Optional alpha channel [N, H, W, 1]
```

### Field Details

#### 1. **images** (Required)
- **Type**: `ImageInput` (torch.Tensor)
- **Shape**: `(N, H, W, 3)` where:
  - N = number of frames
  - H = height in pixels
  - W = width in pixels
  - 3 = RGB channels
- **Value Range**: `[0.0, 1.0]` (float32)
- **Description**: Frame data as normalized floating-point RGB images

#### 2. **frame_rate** (Required)
- **Type**: `Fraction` (from Python's fractions module)
- **Example Values**: 
  - `Fraction(24, 1)` = 24 FPS
  - `Fraction(30000, 1001)` = 29.97 FPS (NTSC)
  - `Fraction(60, 1)` = 60 FPS
- **Description**: The playback speed in frames per second, stored as exact fraction to avoid floating-point precision loss

#### 3. **audio** (Optional, Default: None)
- **Type**: `Optional[AudioInput]` (dict or None)
- **Structure**: `{"waveform": torch.Tensor, "sample_rate": int}`
  - **waveform**: Shape `(1, C, N_samples)` where:
    - 1 = batch dimension (always 1)
    - C = number of audio channels (e.g., 1 for mono, 2 for stereo)
    - N_samples = number of audio samples
  - **sample_rate**: Integer sample rate in Hz (e.g., 44100, 48000)
- **Value Range**: Floating-point PCM samples, typically in [-1.0, 1.0]
- **Description**: Optional audio track synchronized with video frames

#### 4. **metadata** (Optional, Default: None)
- **Type**: `Optional[dict]`
- **Description**: Arbitrary metadata stored with the video (e.g., workflow data, prompt, extra_pnginfo)
- **JSON Serialization**: Values should be JSON-serializable

#### 5. **alpha** (Optional, Default: None)
- **Type**: `Optional[MaskInput]` (torch.Tensor)
- **Shape**: `(N, H, W, 1)`
- **Value Range**: `[0.0, 1.0]` (float32)
- **Description**: Optional per-frame alpha channel for transparency

## VIDEO Type Classes

### VideoInput (Base/Interface)

The abstract base class representing video. All video objects inherit from this:

```python
class VideoInput:
    def get_components(self) -> VideoComponents:
        """Extract all components from the video"""
        pass
    
    def get_dimensions(self) -> tuple[int, int]:
        """Returns (width, height)"""
        pass
    
    def save_to(
        self,
        path: str | io.BytesIO,
        format: VideoContainer = VideoContainer.AUTO,
        codec: VideoCodec = VideoCodec.AUTO,
        metadata: Optional[dict] = None,
    ):
        """Save video to file with optional format/codec conversion"""
        pass
```

### VideoFromComponents

Class for creating VIDEO objects from components (frames, audio, fps):

```python
class VideoFromComponents(VideoInput):
    def __init__(self, components: VideoComponents):
        self.__components = components
    
    def get_components(self) -> VideoComponents:
        return self.__components
    
    def save_to(
        self,
        path: str,
        format: VideoContainer = VideoContainer.AUTO,
        codec: VideoCodec = VideoCodec.AUTO,
        metadata: Optional[dict] = None,
    ):
        # Only supports MP4 format and H264 codec
        pass
```

### VideoFromFile

Class for loading VIDEO from file paths or BytesIO:

```python
class VideoFromFile(VideoInput):
    def __init__(self, file: str | io.BytesIO, *, start_time: float=0, duration: float=0):
        pass
    
    def get_components(self) -> VideoComponents:
        # Decodes video and returns components
        pass
    
    def get_dimensions(self) -> tuple[int, int]:
        # Returns (width, height) from metadata without decoding
        pass
    
    def get_duration(self) -> float:
        # Returns duration in seconds
        pass
    
    def get_frame_rate(self) -> Fraction:
        # Returns FPS as Fraction
        pass
    
    def get_frame_count(self) -> int:
        # Returns total number of frames
        pass
    
    def save_to(...):
        # Can copy-stream or re-encode
        pass
```

## Codec and Container Specifications

### VideoCodec Enum

```python
class VideoCodec(str, Enum):
    AUTO = "auto"   # Auto-detect from source
    H264 = "h264"   # H.264/AVC codec
```

### VideoContainer Enum

```python
class VideoContainer(str, Enum):
    AUTO = "auto"   # Auto-detect from source
    MP4 = "mp4"     # MP4 container (.mp4)
```

**Note**: Only MP4 container with H264 codec is currently fully supported for writing from VideoFromComponents. Other formats require re-encoding through VideoFromFile.

## SaveVideo Node Structure

From `comfy_extras/nodes_video.py`:

```python
class SaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveVideo",
            display_name="Save Video",
            inputs=[
                io.Video.Input("video", tooltip="The video to save."),
                io.String.Input("filename_prefix", default="video/ComfyUI"),
                io.Combo.Input("format", options=["auto", "mp4"], default="auto"),
                io.Combo.Input("codec", options=["auto", "h264"], default="auto"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )
    
    @classmethod
    def execute(cls, video: Input.Video, filename_prefix, format: str, codec) -> io.NodeOutput:
        width, height = video.get_dimensions()
        # ... file path handling ...
        video.save_to(
            output_path,
            format=Types.VideoContainer(format),
            codec=codec,
            metadata=saved_metadata
        )
```

## Example: Creating a VIDEO Object

```python
from fractions import Fraction
import torch
from comfy_api.latest._input_impl.video_types import VideoFromComponents
from comfy_api.latest._util.video_types import VideoComponents

# Create frames: (N_frames, height, width, 3)
frames = torch.rand(30, 1080, 1920, 3)  # 30 frames of 1080p

# Optional: Create audio
audio = {
    "waveform": torch.randn(1, 2, 48000),  # 1 second of stereo audio at 48kHz
    "sample_rate": 48000
}

# Create VideoComponents
components = VideoComponents(
    images=frames,
    frame_rate=Fraction(30, 1),  # 30 FPS
    audio=audio,
    metadata={"prompt": "some workflow data"}
)

# Create VIDEO object
video = VideoFromComponents(components)

# Save to file
video.save_to(
    "/path/to/output.mp4",
    format="auto",
    codec="auto",
    metadata={"custom_key": "custom_value"}
)
```

## Current Implementation vs. Expected VIDEO Type

### In openrouter_shared.py

Currently, `node_video_gen.py` returns **IMAGE** type instead of VIDEO:

```python
RETURN_TYPES = ("IMAGE", "STRING", "STRING")
RETURN_NAMES = ("video", "Stats", "Credits")
```

The `video_bytes_to_frames()` function returns:
```python
def video_bytes_to_frames(video_bytes, extract_fps=True):
    # ... decoding ...
    return frames_tensor, metadata
    # Returns: (torch.Tensor[N, H, W, 3], dict with fps, frame_count, etc.)
```

### Required Changes to Match SaveVideo Expectations

1. **Change return type from IMAGE to VIDEO** in node_video_gen.py
2. **Create VideoComponents object** with:
   - `images`: The decoded frames tensor
   - `frame_rate`: Fraction from the metadata fps
   - `audio`: Optional (can be None for video-only)
   - `metadata`: The video metadata dict
3. **Wrap in VideoFromComponents** to create the proper VIDEO type
4. **Return VIDEO type** that SaveVideo can consume

### Conversion Code Needed

```python
from fractions import Fraction
from comfy_api.latest._input_impl.video_types import VideoFromComponents
from comfy_api.latest._util.video_types import VideoComponents

def generate_video(self, ...):
    # ... existing code ...
    frames_tensor, video_metadata = shared.video_bytes_to_frames(video_bytes)
    
    # Convert to VIDEO type
    components = VideoComponents(
        images=frames_tensor,
        frame_rate=Fraction(int(video_metadata["fps"]), 1),
        audio=None,  # OpenRouter doesn't provide audio currently
        metadata=video_metadata
    )
    video = VideoFromComponents(components)
    
    return (video, stats, credits)
```

## Key Differences from AUDIO Type

| Aspect | AUDIO | VIDEO |
|--------|-------|-------|
| Structure | Simple dict with 2 keys | Composite VideoComponents dataclass |
| Frame Data | Single waveform array | Tensor of N frames |
| Metadata | Not included in AUDIO dict | Included in VideoComponents |
| Container | Dict | Class instance with methods |
| Methods | None | get_components(), get_dimensions(), save_to() |
| Node Return Pattern | Tuple of dicts | Tuple of VideoInput objects |

## Summary

**A VIDEO type is NOT just frames as IMAGE.** It is a sophisticated composite type that bundles:
1. Frame tensor (images)
2. Precise frame rate as Fraction
3. Optional audio track
4. Optional metadata
5. Instance methods for inspection and saving

When passing VIDEO between nodes (especially to SaveVideo), the receiving node:
- Calls `video.get_components()` to extract components
- Can access `video.get_dimensions()` for width/height
- Can call `video.save_to()` to write to disk with codec/format control

Current code returns bare IMAGE tensors. To be compatible with SaveVideo, it must return a proper VideoFromComponents object wrapping VideoComponents.
