import os
from dataclasses import dataclass

@dataclass
class StoryConfig:
    """
    故事配置类，使用dataclass来管理故事的基本参数
    
    属性：
        language: 故事语言（默认中文）
        words_per_paragraph: 每段字数（默认68字）
        target_age: 目标年龄段（默认5岁）
        paragraph_count: 段落数量（默认12段）
    """
    language: str = os.getenv("STORY_LANGUAGE", "中文")
    words_per_paragraph: int = int(os.getenv("WORDS_PER_PARAGRAPH", "68"))
    target_age: str = os.getenv("TARGET_AGE", "5岁")
    paragraph_count: int = int(os.getenv("PARAGRAPH_COUNT", "12"))