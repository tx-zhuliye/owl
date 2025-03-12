# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import ffmpeg
from PIL import Image
from scenedetect import (  # type: ignore[import-untyped]
    SceneManager,
    VideoManager,
)
from scenedetect.detectors import (  # type: ignore[import-untyped]
    ContentDetector,
)

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import OpenAIAudioModels, BaseModelBackend
from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.utils import dependencies_required

from .video_downloader_toolkit import (
    VideoDownloaderToolkit,
    _capture_screenshot,
)

logger = logging.getLogger(__name__)

VIDEO_QA_PROMPT = """
Analyze the provided video frames and corresponding audio transcription to \
answer the given question(s) thoroughly and accurately.

Instructions:
    1. Visual Analysis:
        - Examine the video frames to identify visible entities.
        - Differentiate objects, species, or features based on key attributes \
such as size, color, shape, texture, or behavior.
        - Note significant groupings, interactions, or contextual patterns \
relevant to the analysis.

    2. Audio Integration:
        - Use the audio transcription to complement or clarify your visual \
observations.
        - Identify names, descriptions, or contextual hints in the \
transcription that help confirm or refine your visual analysis.

    3. Detailed Reasoning and Justification:
        - Provide a brief explanation of how you identified and distinguished \
each species or object.
        - Highlight specific features or contextual clues that informed \
your reasoning.

    4. Comprehensive Answer:
        - Specify the total number of distinct species or object types \
identified in the video.
        - Describe the defining characteristics and any supporting evidence \
from the video and transcription.

    5. Important Considerations:
        - Pay close attention to subtle differences that could distinguish \
similar-looking species or objects 
          (e.g., juveniles vs. adults, closely related species).
        - Provide concise yet complete explanations to ensure clarity.

**Audio Transcription:**
{audio_transcription}

**Question:**
{question}
"""


class VideoAnalysisToolkit(BaseToolkit):
    r"""A class for analysing videos with vision-language model.

    Args:
        download_directory (Optional[str], optional): The directory where the
            video will be downloaded to. If not provided, video will be stored
            in a temporary directory and will be cleaned up after use.
            (default: :obj:`None`)
    """

    @dependencies_required("ffmpeg", "scenedetect")
    def __init__(
        self,
        download_directory: Optional[str] = None,
        model: Optional[BaseModelBackend] = None,
    ) -> None:
        self._cleanup = download_directory is None

        self._download_directory = Path(
            download_directory or tempfile.mkdtemp()
        ).resolve()

        self.video_downloader_toolkit = VideoDownloaderToolkit(
            download_directory=str(self._download_directory)
        )

        try:
            self._download_directory.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            raise ValueError(
                f"{self._download_directory} is not a valid directory."
            )
        except OSError as e:
            raise ValueError(
                f"Error creating directory {self._download_directory}: {e}"
            )

        logger.info(f"Video will be downloaded to {self._download_directory}")

        self.vl_model = model

        self.vl_agent = ChatAgent(
            model=self.vl_model, output_language="English"
        )
        try:
            self.audio_models = OpenAIAudioModels()
        except Exception as e:
            print('fallback to whisper model')

    def convert_webm_to_mp4(self, video_path: str) -> str:
        r"""Convert a .webm video file to .mp4 format.

        Args:
            video_path (str): The path to the .webm video file.

        Returns:
            str: The path to the converted .mp4 file.

        Raises:
            RuntimeError: If the conversion fails.
        """
        import os
        # 检查文件是否存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # 检查文件是否为 .webm 格式
        if not video_path.lower().endswith(".webm"):
            raise ValueError(f"File is not a .webm video: {video_path}")

        # 生成输出文件路径
        output_path = video_path.rsplit('.', 1)[0] + ".mp4"

        # 如果输出文件已存在，直接返回
        if os.path.exists(output_path):
            return output_path

        try:
            # 使用 FFmpeg 转换格式
            (
                ffmpeg.input(video_path)
                .output(output_path, vcodec="libx264", acodec="aac")
                .run()
            )
            return output_path
        except ffmpeg.Error as e:
            raise RuntimeError(f"FFmpeg-Python failed: {e}")

    def split_audio_segments(self, video_path, segment_duration=60 * 2, sample_rate=16000):
        """
        将视频中的音频拆分为多个大小相同的片段，并保存为 WAV 格式。

        Args:
            video_path (str): 视频文件路径。
            output_dir (str): 输出目录路径。
            segment_duration (int): 每个音频片段的时长（秒）。
            sample_rate (int): 音频采样率（Hz）。

        Returns:
            List[str]: 生成的音频片段文件路径列表。
        """
        import os
        # 确保输出目录存在
        output_dir = os.path.dirname(video_path)

        # 生成输出文件模板
        output_template = os.path.join(output_dir, "segment_%03d.wav")

        # 使用 FFmpeg 拆分音频
        try:
            (
                ffmpeg.input(video_path)
                .output(
                    output_template,
                    f="segment",  # 使用 segment 过滤器
                    segment_time=segment_duration,  # 每个片段的时长
                    acodec="pcm_s16le",  # 使用 WAV 格式的编码器
                    ar=sample_rate,  # 设置采样率
                    ac=1,  # 单声道（可选）
                )
                .run()
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"FFmpeg failed: {e}")

        # 返回生成的音频片段文件列表
        return sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith("segment_")])

    def _extract_audio_from_video(
        self, video_path: str, output_format: str = "mp3"
    ) -> str:
        r"""Extract audio from the video.

        Args:
            video_path (str): The path to the video file.
            output_format (str): The format of the audio file to be saved.
                (default: :obj:`"mp3"`)

        Returns:
            str: The path to the audio file."""

        output_path = video_path.rsplit('.', 1)[0] + f".{output_format}"
        import os
        if os.path.exists(output_path):
            return output_path
        
        try:
            (
                ffmpeg.input(video_path)
                .output(output_path, vn=None, acodec="libmp3lame")
                .run()
            )
            return output_path
        except ffmpeg.Error as e:
            raise RuntimeError(f"FFmpeg-Python failed: {e}")

    def _transcribe_audio(self, segments: list[str]) -> str:
        r"""Transcribe the audio of the video."""
        try:
            result = ''
            for segment in segments:
                audio_transcript = self.audio_models.speech_to_text(segment)
                result += audio_transcript
            return result
        except Exception as e:
            audio_transcript = self.transcribe_audio_whisper(segments=segments)
        return audio_transcript
    
    def check_video_corruption(self, video_path: str) -> bool:
        r"""Check if a video file is corrupted.

        Args:
            video_path (str): The path to the video file.

        Returns:
            bool: True if the video is not corrupted, False otherwise.
        """
        try:
            # 使用 FFmpeg 检查文件是否可读取
            ffmpeg.probe(video_path)
            return True
        except ffmpeg.Error as e:
            print(f"Video file is corrupted or cannot be read: {video_path}")
            print(f"Error: {e.stderr.decode()}")
            return False
    

    def transcribe_audio_whisper(self, segments, type='tiny'):
        import whisper
        from concurrent.futures import ProcessPoolExecutor
        import os
        import time

        # 设置 OpenMP 为单线程模式
        os.environ["OMP_NUM_THREADS"] = "1"

        # 加载 Whisper 模型
        start = time.time()
          # 显式使用 FP32

        def transcribe_segment(segment_path):
            # 禁用 FFmpeg 日志输出
            import whisper
            # whisper.transcribe = whisper.transcribe.DisableFFmpegLogging()
            model = whisper.load_model(type, device="cpu")
            # 转录音频片段
            result = model.transcribe(segment_path)
            return result['text']

        # 使用线程池并行转录
        results = []
        for segment_path in segments:
            result = transcribe_segment(segment_path)
            results.append(result)

        print(f'audio to txt cost {time.time() - start}')
        return results

    def _extract_keyframes(
        self, video_path: str, num_frames: int, threshold: float = 25.0
    ) -> List[Image.Image]:
        r"""Extract keyframes from a video based on scene changes
        and return them as PIL.Image.Image objects.

        Args:
            video_path (str): Path to the video file.
            num_frames (int): Number of keyframes to extract.
            threshold (float): The threshold value for scene change detection.

        Returns:
            list: A list of PIL.Image.Image objects representing
                the extracted keyframes.
        """
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))

        video_manager.set_duration()
        video_manager.start()
        scene_manager.detect_scenes(video_manager)

        scenes = scene_manager.get_scene_list()
        keyframes: List[Image.Image] = []

        for start_time, _ in scenes:
            if len(keyframes) >= num_frames:
                break
            frame = _capture_screenshot(video_path, start_time)
            keyframes.append(frame)

        print(len(keyframes))
        return keyframes

    def ask_question_about_video(
        self,
        video_path: str,
        question: str,
        num_frames: int = 28,
        # 28 is the maximum number of frames
        # that can be displayed in a single message for
        # the Qwen-VL-Max model
    ) -> str:
        r"""Ask a question about the video.

        Args:
            video_path (str): The path to the video file.
                It can be a local file or a URL (such as Youtube website).
            question (str): The question to ask about the video.
            num_frames (int): The number of frames to extract from the video.
                To be adjusted based on the length of the video.
                (default: :obj:`28`)

        Returns:
            str: The answer to the question.
        """

        from urllib.parse import urlparse

        parsed_url = urlparse(video_path)
        is_url = all([parsed_url.scheme, parsed_url.netloc])

        if is_url:
            video_path = self.video_downloader_toolkit.download_video(
                video_path
            )
        
        if video_path.lower().endswith(".webm"):
            video_path = self.convert_webm_to_mp4(video_path=video_path)
        video_frames = self._extract_keyframes(video_path, num_frames)
        segments = self.split_audio_segments(video_path)
        audio_transcript = self._transcribe_audio(segments)

        prompt = VIDEO_QA_PROMPT.format(
            audio_transcription=','.join(audio_transcript),
            question=question,
        )

        print(prompt)

        msg = BaseMessage.make_user_message(
            role_name="User",
            content=prompt,
            image_list=video_frames,
        )

        response = self.vl_agent.step(msg)
        answer = response.msgs[0].content

        return answer

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the
        functions in the toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects representing
                the functions in the toolkit.
        """
        return [FunctionTool(self.ask_question_about_video)]