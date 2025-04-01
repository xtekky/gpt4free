from datetime import datetime
from typing import Dict, List, Optional

class PromptQualityChecker:
    """提示词质量检查器类
    负责验证和优化生成的提示词质量
    """
    
    def __init__(self):
        """初始化质量检查器"""
        # 必需的关键词列表
        self.required_style_keywords = [
            "children's book illustration",
            "digital art",
            "masterpiece",
            "best quality",
            "highly detailed"
        ]
        
        # 禁用的关键词列表
        self.forbidden_keywords = [
            "nsfw",
            "ugly",
            "scary",
            "horror",
            "violent",
            "blood",
            "gore",
            "disturbing"
        ]
        
        # 场景元素检查列表
        self.scene_elements = {
            "environment": ["background", "setting", "scene"],
            "lighting": ["light", "shadow", "illumination"],
            "mood": ["atmosphere", "mood", "feeling"],
            "composition": ["composition", "layout", "arrangement"]
        }
    
    def check_prompt_completeness(self, prompt: str) -> tuple:
        """检查提示词是否包含所有必需的元素
        
        参数：
            prompt: 待检查的提示词
            
        返回：
            tuple: (是否通过检查, 缺失元素列表)
        """
        missing_elements = []
        prompt_lower = prompt.lower()
        
        # 检查必需的风格关键词
        for keyword in self.required_style_keywords:
            if keyword.lower() not in prompt_lower:
                missing_elements.append(f"Missing style keyword: {keyword}")
        
        # 检查场景元素
        for category, elements in self.scene_elements.items():
            if not any(element in prompt_lower for element in elements):
                missing_elements.append(f"Missing {category} description")
        
        return len(missing_elements) == 0, missing_elements
    
    def check_forbidden_content(self, prompt: str) -> tuple:
        """检查提示词是否包含禁用内容
        
        参数：
            prompt: 待检查的提示词
            
        返回：
            tuple: (是否通过检查, 发现的禁用词列表)
        """
        found_forbidden = []
        prompt_lower = prompt.lower()
        
        for keyword in self.forbidden_keywords:
            if keyword.lower() in prompt_lower:
                found_forbidden.append(keyword)
        
        return len(found_forbidden) == 0, found_forbidden
    
    def validate_character_balance(self, prompt: str, character_weights: Dict[str, float]) -> tuple:
        """验证角色描述的平衡性
        
        参数：
            prompt: 待检查的提示词
            character_weights: 角色权重字典
            
        返回：
            tuple: (是否平衡, 不平衡原因)
        """
        prompt_lower = prompt.lower()
        character_mentions = {}
        
        # 统计每个角色在提示词中的出现次数
        for character_name in character_weights.keys():
            character_mentions[character_name] = prompt_lower.count(character_name.lower())
        
        # 检查是否有角色完全未被提及
        for name, mentions in character_mentions.items():
            if mentions == 0:
                return False, f"Character {name} is not mentioned in the prompt"
        
        # 检查提及次数是否与权重基本成比例
        total_mentions = sum(character_mentions.values())
        if total_mentions == 0:
            return False, "No characters are mentioned in the prompt"
            
        for name, weight in character_weights.items():
            expected_mentions = total_mentions * weight
            actual_mentions = character_mentions[name]
            
            # 允许20%的误差
            if abs(actual_mentions - expected_mentions) > expected_mentions * 0.2:
                return False, f"Character {name} mention count ({actual_mentions}) does not match its weight ({weight:.2f})"
        
        return True, ""
    
    def enhance_prompt(self, prompt: str) -> str:
        """增强提示词质量
        
        参数：
            prompt: 原始提示词
            
        返回：
            str: 增强后的提示词
        """
        # 确保包含所有必需的风格关键词
        for keyword in self.required_style_keywords:
            if keyword.lower() not in prompt.lower():
                prompt += f", {keyword}"
        
        # 添加质量控制关键词
        quality_keywords = [
            "professional lighting",
            "perfect composition",
            "vivid colors",
            "smooth lines"
        ]
        
        for keyword in quality_keywords:
            if keyword.lower() not in prompt.lower():
                prompt += f", {keyword}"
        
        return prompt