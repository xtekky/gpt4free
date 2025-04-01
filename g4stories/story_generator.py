import os
import json
from g4f.client import Client
from .story_config import StoryConfig

class StoryGenerator:
    def __init__(self):
        """初始化故事生成器，设置系统提示词"""
        # 从环境变量获取模型名称和API配置
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.system_prompt = """你是一位专业的儿童故事作家。
            你的任务是创作适合儿童阅读的有趣故事。

            ### 要求：
            1. 故事要有教育意义和趣味性
            2. 语言要简单易懂，适合目标年龄段
            3. 情节要生动有趣，富有想象力
            4. 要传递正面的价值观
            5. 要符合儿童认知水平
            6. 要有清晰的故事结构和情节发展

            ### 故事类型指南：
            1. **冒险故事**
            - 设定明确的冒险目标
            - 创造适度的挑战和困难
            - 展现解决问题的智慧
            - 体现团队合作精神
            - 注重安全意识的传达

            2. **生活故事**
            - 选择贴近儿童生活的场景
            - 描写真实的情感体验
            - 融入生活智慧和技能
            - 培养良好的生活习惯
            - 展现积极的处事态度

            3. **童话故事**
            - 创造独特的奇幻元素
            - 设计富有想象力的情节
            - 融入简单易懂的寓意
            - 平衡现实与想象的关系
            - 传递美好的价值观念

            ### 角色塑造指南：
            1. **主角设计**
            - 设定鲜明的性格特征
            - 赋予独特的兴趣爱好
            - 创造合理的成长空间
            - 设计可爱的外形特征
            - 突出积极的品格特点

            2. **配角设计**
            - 明确与主角的关系
            - 设计互补的性格特征
            - 创造独特的个性魅力
            - 合理安排出场比重
            - 体现群体的多样性

            ### 情节设计指南：
            1. **开头设计**
            - 简洁有趣的背景介绍
            - 吸引人的开场方式
            - 明确的故事起因
            - 自然的人物登场
            - 营造轻松的氛围

            2. **发展过程**
            - 循序渐进的情节推进
            - 适度的悬念设置
            - 合理的冲突安排
            - 丰富的情节细节
            - 注重因果关系

            3. **结局设计**
            - 圆满的问题解决
            - 温暖的情感升华
            - 明确的教育意义
            - 留下思考空间
            - 激发积极行动

            ### 对话设计指南：
            1. **语言特点**
            - 使用简单易懂的词汇
            - 保持语言的趣味性
            - 适当运用拟声词
            - 控制句子长度
            - 注意语气的变化

            2. **对话功能**
            - 展现人物性格
            - 推动情节发展
            - 传递故事主题
            - 增添趣味性
            - 体现情感交流

            ### 教育价值设计：
            1. **知识传递**
            - 自然融入知识点
            - 设置互动学习环节
            - 培养观察思考能力
            - 激发探索兴趣
            - 鼓励创造思维

            2. **品德培养**
            - 融入正确价值观
            - 培养同理心
            - 鼓励勇气和担当
            - 强调诚实守信
            - 重视团队合作"""
        self.StoryConfig= StoryConfig
        
    def generate_story(self, 
                    theme: str,
                    config: StoryConfig,
                    additional_requirements:None):
        """
            生成儿童故事的核心方法
            
            参数：
                theme: 故事主题
                config: 故事配置对象
                additional_requirements: 额外的故事要求
                
            返回：
                包含完整故事内容的字典，包括标题、角色描述、段落内容等
        """
        try:
            # 构建完整的提示词，包含所有故事生成要求
            prompt = f"""请为{config.target_age}的儿童创作一个关于{theme}的绘本故事。

## 基本要求：
1. 故事语言为{config.language}
2. 每段约{config.words_per_paragraph}字
3. 共{config.paragraph_count}段
4. 适合{config.target_age}儿童阅读

## 故事结构要求：
1. 清晰的开端、发展、高潮、结局
2. 情节连贯，富有想象力
3. 角色形象鲜明
4. 结尾要留有适当的想象空间
5. 确保故事传递积极正面的价值观

## 输出格式：
请将故事内容格式化为以下JSON格式：
{{\\"title\\": \\"故事标题\\",\\"characters\\": [\\"角色1描述\\",\\"角色2描述\\"],\\"paragraphs\\": [\\"第一段内容\\",\\"第二段内容\\"]}}"""
            if additional_requirements:
                prompt += f"\n\n## 额外要求：\n{additional_requirements}"
            client = Client()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                web_search=False
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
                story_content = json.loads(content)
                if not isinstance(story_content, dict):
                    raise ValueError("Response is not a dictionary")
                
                # 验证必要的字段
                required_fields = ["title", "characters", "paragraphs"]
                for field in required_fields:
                    if field not in story_content:
                        raise ValueError(f"Missing required field: {field}")
                
                # 验证字段类型
                if not isinstance(story_content["title"], str):
                    raise ValueError("Title must be a string")
                if not isinstance(story_content["characters"], list):
                    raise ValueError("Characters must be a list")
                if not isinstance(story_content["paragraphs"], list):
                    raise ValueError("Paragraphs must be a list")
                
                # 验证内容不为空
                if not story_content["title"].strip():
                    raise ValueError("Title cannot be empty")
                if not story_content["characters"]:
                    raise ValueError("Characters list cannot be empty")
                if not story_content["paragraphs"]:
                    raise ValueError("Paragraphs list cannot be empty")
                
                print(f"成功生成故事：{story_content['title']}")
                return story_content

            except json.JSONDecodeError:
                print(f"JSON解析错误。API返回内容：\n{content}")
                return None
            except ValueError as e:
                print(f"内容格式错误: {str(e)}")
                return None
        except Exception as e:
            print(f"生成故事时发生错误: {str(e)}")
            if hasattr(e, 'response'):
                print(f"API响应: {e.response}")
            return None

