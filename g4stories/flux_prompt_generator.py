import os
import json
from typing import Dict
from .prompt_quality_checker import PromptQualityChecker
from g4f.client import Client

class FluxPromptGenerator:
    """
    Flux提示词生成器类
    负责为故事场景生成AI绘图提示词
    """
    
    def __init__(self):
        """初始化提示词生成器，设置系统提示词"""
        # 从环境变量获取模型名称和API配置
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # 添加角色特征历史记录
        self.character_history = {}
        
        # 设置系统提示词
        self.system_prompt = """你是一位专业的儿童绘本插画提示词工程师。
你的任务是为儿童故事场景生成高质量的Flux AI绘图提示词。

### 角色一致性维护指南：
1. **角色特征库管理**
   - 创建并维护所有角色的特征数据库
   - 记录每个角色（包括主角和配角）的关键特征
   - 确保特征描述的准确性和可复现性

2. **跨场景一致性检查**
   - 在生成新场景前，查阅所有角色的历史形象
   - 确保场景中每个角色的描述与其历史一致
   - 对角色特征的任何改变需要合理的剧情支持

3. **提示词模板系统**
   - 为所有角色建立基础提示词模板
   - 模板中固定每个角色的关键特征描述
   - 只允许改变与情节相关的动作和表情描述

4. **质量控制措施**
   - 建立角色一致性评估标准
   - 定期检查生成图片的角色一致性
   - 收集和分析不一致案例，优化提示词模板

### 场景平衡指南：
1. **角色权重分配**
   - 根据场景描述确定所有出场角色
   - 合理分配画面空间和注意力
   - 确保配角得到适当的视觉表现

2. **互动描述要求**
   - 详细描述角色之间的互动关系
   - 明确指定各个角色的位置和动作
   - 避免角色被动或缺乏存在感

3. **场景构图建议**
   - 根据剧情重要性安排角色位置
   - 使用构图技巧突出重要互动
   - 保持画面的视觉平衡

### 输出要求：
1. 提示词必须使用英文
2. 提示词要准确描述场景和所有人物
3. 提示词要符合儿童绘本的温馨可爱风格
4. 角色描述要求：
   - 为每个出场角色创建详细的特征描述
   - 描述需包含：
     * 外貌特征：身高、体型、肤色、发型、标志性特征
     * 服装风格：日常装扮、颜色搭配、特殊配饰
     * 表情习惯：常见表情、特殊表情、情绪表达方式
     * 姿态特点：站姿、行走方式、标志性动作
   - 每次生成场景时，参考所有角色的特征模板
   - 使用固定的角色描述词，确保形象一致性
   - 对于角色的不同表情和动作，保持基础特征不变
5. 输出必须是JSON格式，包含以下字段：
   - Title: 场景标题
   - Characters: 场景中所有角色及其特征描述
   - Positive Prompt: 正向提示词，包含：
     * 场景整体描述 (Scene Overview)
     * 角色互动描述 (Character Interactions)
     * 每个角色的具体描述 (Character Descriptions)
     * 艺术风格 (Art Style): children's book illustration, digital art, cute, warm
     * 画面质量 (Quality): masterpiece, best quality, highly detailed
     * 光照效果 (Lighting): soft lighting, warm colors
   - Negative Prompt: 负向提示词，用于避免不需要的元素：
     * 通用负向词: nsfw, ugly, duplicate, morbid, mutilated, poorly drawn face
     * 画面控制: blurry, bad anatomy, bad proportions, extra limbs, text, watermark
     * 风格控制: photo-realistic, 3d render, cartoon, anime, sketches
     * 角色一致性控制: inconsistent character features, varying character design

### 示例输出：
```json
{
    "Title": "Lily and Tom's Adventure",
    "Characters": {
        "Lily": {
            "role": "main",
            "base_features": "A 7-year-old girl with shoulder-length curly brown hair, round face, bright green eyes, and a small heart-shaped birthmark on her right cheek",
            "clothing": "Light blue overall dress with white polka dots, yellow t-shirt underneath, red canvas shoes",
            "accessories": "Rainbow hair clips, silver heart-shaped locket necklace",
            "expressions": "Wide bright smile showing slightly gapped front teeth, dimples when smiling"
        },
        "Tom": {
            "role": "supporting",
            "base_features": "A 6-year-old boy with short black curly hair, warm brown eyes, and freckles across his nose",
            "clothing": "Green striped t-shirt, blue denim shorts, white sneakers with yellow laces",
            "accessories": "Red baseball cap worn slightly tilted, silver robot-shaped pendant",
            "expressions": "Curious eyes and enthusiastic grin showing missing front tooth"
        }
    },
    "Positive Prompt": "A heartwarming scene in a sunny park with Lily and Tom building a sandcastle together. Lily (7-year-old girl, shoulder-length curly brown hair, round face, bright green eyes, heart-shaped birthmark, wearing light blue polka dot overall dress) is carefully decorating the castle's tower with small shells, showing her characteristic dimpled smile. Tom (6-year-old boy, short black curly hair, freckles, wearing green striped t-shirt and red cap) kneels on the other side, excitedly adding a moat around the castle, his eyes sparkling with creativity. children's book illustration style, digital art, masterpiece, best quality, highly detailed, soft lighting, warm colors, peaceful atmosphere",
    "Negative Prompt": "nsfw, ugly, duplicate, morbid, mutilated, poorly drawn face, blurry, bad anatomy, bad proportions, extra limbs, text, watermark, photo-realistic, 3d render, cartoon, anime, sketches, inconsistent character features, varying character design"
}
```"""


    def generate_prompts(self, title: str, scene: str, main_character: str) -> Dict:
        """
        为场景生成图像提示词
        
        参数：
            title: 故事标题
            scene: 场景描述
            main_character: 主角名称
            
        返回：
            包含正向和负向提示词的字典
        """
        try:
            # 构建提示词生成的请求
            prompt = f"""请为以下儿童故事场景生成Flux绘图提示词：

故事标题：{title}
场景描述：{scene}
主角：{main_character}

要求：
1. 正向提示词必须包含场景描述、主角特征、艺术风格、画面质量和光照效果
2. 负向提示词必须包含所有必要的控制词
3. 确保输出格式为规定的JSON格式
4. 所有提示词必须是英文
5. 风格必须是儿童绘本插画风格"""
            
            # 调用OpenAI API生成提示词
            client = Client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )

            # 获取生成的内容
            content = response.choices[0].message.content.strip()
            
            try:
                # 如果返回的内容被包裹在```json和```中，去掉这些标记
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                
                # 尝试解析JSON
                prompt_content = json.loads(content)
                
                # 验证必要的字段
                required_fields = ["Title", "Positive Prompt", "Negative Prompt"]
                for field in required_fields:
                    if field not in prompt_content:
                        raise ValueError(f"Missing required field: {field}")
                
                # 验证字段类型和内容
                if not isinstance(prompt_content["Title"], str):
                    raise ValueError("Title must be a string")
                if not isinstance(prompt_content["Positive Prompt"], str):
                    raise ValueError("Positive Prompt must be a string")
                if not isinstance(prompt_content["Negative Prompt"], str):
                    raise ValueError("Negative Prompt must be a string")
                
                # 进行质量检查和增强
                quality_checker = PromptQualityChecker()
                is_complete, missing_elements = quality_checker.check_prompt_completeness(
                    prompt_content["Positive Prompt"]
                )
                if not is_complete:
                    print(f"Warning: Prompt is missing elements: {missing_elements}")
                    prompt_content["Positive Prompt"] = quality_checker.enhance_prompt(
                        prompt_content["Positive Prompt"]
                    )
                
                is_clean, forbidden_words = quality_checker.check_forbidden_content(
                    prompt_content["Positive Prompt"]
                )
                if not is_clean:
                    print(f"Warning: Prompt contains forbidden words: {forbidden_words}")
                
                return {
                    "positive_prompt": prompt_content["Positive Prompt"],
                    "negative_prompt": prompt_content["Negative Prompt"]
                }
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                return None
            except ValueError as e:
                print(f"Error validating prompt content: {e}")
                return None
            
        except Exception as e:
            print(f"Error generating prompts: {e}")
            return None