import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from .flux_image_generator import FluxImageGenerator
from .flux_prompt_generator import FluxPromptGenerator
from g4f.client import Client

class StoryFormatter:
    """
    故事格式化器类
    负责将生成的故事转换为markdown格式，并添加词汇解释
    """
    
    def __init__(self):
        """初始化格式化器，设置系统提示词"""
        # 从环境变量获取模型名称和API配置
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.system_prompt = """Role: 儿童故事绘本排版整理专家

## Profile
- **Author:** 玄清工作流
- **Description:** 专注于将儿童故事绘本转化为适合儿童阅读的Markdown格式，保持原文内容和图片链接不变，优化排版以提升阅读体验和理解效果。

## Attention
1. **内容完整性：** 不修改故事内容的顺序和文本，确保原始故事的连贯性和完整性。
2. **图片保留：** 不删除任何图片链接及相关内容，确保视觉元素完整呈现。
3. **排版优化：** 仅优化排版，提升文本的可读性，并在故事结尾添加文中难点词汇的尾注，提供词汇的解释和翻译，辅助儿童理解。

## Goals
1. **格式优化：** 将儿童故事绘本的文本排版优化为符合儿童阅读习惯的Markdown格式，确保结构清晰。
2. **内容保持：** 完整保留文本内容、图片链接及所有原始元素，不做任何内容删减或修改。
3. **提升阅读体验：** 设计简洁、友好的版面布局，通过合理的排版和视觉元素，增强儿童的阅读兴趣和理解能力。

## Skills
1. **Markdown专业知识：** 深入掌握Markdown排版规范，能够灵活运用各种Markdown元素，如标题、列表、图片嵌入和代码块等，创建结构化且美观的文档。
2. **语言与词汇处理：** 擅长识别和提取文本中的难点词汇，能够准确理解其上下文含义，并在尾注中提供简明易懂的解释和翻译，帮助儿童扩展词汇量。
3. **儿童友好设计：** 具备设计简洁、直观的排版能力，能够根据儿童的阅读习惯和认知特点，优化文本布局和视觉呈现，确保内容既吸引人又易于理解。
4. **细节审查能力：** 具备高度的细致性，能够仔细检查文本和图片链接的准确性，确保最终输出的文档无误且高质量。

## Constraints
1. **内容不变：** 严格不修改故事文本的内容或顺序，确保所有原始内容完整保留，不做任何删减或调整。
2. **语种一致：** 输出文档必须与输入内容保持相同的语言，不进行任何语言转换或混用。
3. **真实呈现：** 禁止随意编造内容，所有输出内容必须基于用户提供的原始故事文本，确保真实性和一致性。
4. **标准格式：** 确保文档遵循标准的Markdown语法规范，版面设计需符合儿童阅读的视觉需求，保持整洁和易读性。

## Output Format
1. **标题：** 使用`#`标题格式，确保标题清晰且易于识别。
2. **正文：** 段落保持简短，每个段落之间用空行分隔。适当使用无序列表、加粗或斜体突出重点词汇，帮助儿童理解，同时保留所有图片。
3. **分隔符：** 文章与尾注之间使用水平分隔线`---`，清晰区分内容主体与解释部分。
4. **尾注：** 列出难解词汇，词汇后附上简短易懂的翻译或注释，帮助儿童理解故事内容。

## Workflows
1. **读取故事文本**
   - 获取并导入儿童故事文本
   - 确保所有图片链接完整无误
   - 验证文本格式，确保无乱码或缺失内容

2. **进行Markdown格式化**
   - 将章节标题转换为相应的Markdown标题格式（使用`#`）
   - 将段落内容转换为Markdown段落，保持段落简洁
   - 插入图片链接，确保语法正确并图片显示正常
   - 标记对话内容为引用或特定格式，以增强可读性
   - 识别并收集文本中的难点词汇，准备添加到尾注

3. **排版优化**
   - 调整标题层级，确保结构清晰且逻辑分明
   - 使用无序列表、加粗或斜体等Markdown元素突出重点词汇
   - 确保图片位置合理，避免破坏文本流畅性
   - 调整段落间距，提高整体可读性和视觉舒适度
   - 确保使用一致的字体样式和大小，符合儿童阅读习惯

4. **文档检查**
   - 校对文本内容，确保无拼写或语法错误
   - 检查所有图片链接的有效性，确保图片能正确显示
   - 确认尾注内容的准确性和完整性，确保词汇解释清晰易懂
   - 进行最终预览，确保整体布局适合儿童阅读，版面整洁美观"""

    def process_story(self, story_content: Dict, output_dir: Path) -> Optional[str]:
        """
        处理故事内容，生成最终的故事文件
        
        参数：
            story_content: 故事内容字典
            output_dir: 输出目录
            
        返回：
            str: 生成的故事文件路径，如果失败则返回None
        """
        try:
            # 准备输出目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            story_file = output_dir / f"{story_content['title']}_{timestamp}.md"
            
            # 创建图片目录
            images_dir = output_dir.parent / "generated_images"
            images_dir.mkdir(parents=True, exist_ok=True)
            
            # 初始化图片生成器
            image_generator = FluxImageGenerator()
            
            # 初始化提示词生成器
            prompt_generator = FluxPromptGenerator()
            
            # 生成Markdown格式的故事内容
            markdown_content = [
                f"# {story_content['title']}\n",
                "**角色：**"
            ]
            
            # 添加角色描述
            for character in story_content['characters']:
                markdown_content.append(f"- {character}")
            
            markdown_content.append("\n---\n")
            
            # 处理每个段落
            for i, paragraph in enumerate(story_content['paragraphs']):
                # 添加段落内容
                markdown_content.append(paragraph)
                
                # 为段落生成图片
                scene_name = f"scene_{i+1}"
                image_path = images_dir / f"{story_content['title']}_{scene_name}.png"
                
                # 生成图片的提示词
                prompts = prompt_generator.generate_prompts(
                    title=story_content['title'],
                    scene=paragraph,
                    main_character=story_content.get('main_character', '')
                )
                
                if prompts:
                    # 生成图片
                    success = image_generator.generate_image(
                        positive_prompt=prompts['positive_prompt'],
                        negative_prompt=prompts['negative_prompt'],
                        output_path=str(image_path)
                    )
                    
                    if success:
                        # 使用相对路径添加图片链接
                        rel_path = os.path.relpath(image_path, output_dir)
                        markdown_content.append(f"\n![{scene_name}]({rel_path})\n")
                    else:
                        print(f"生成图片失败: {scene_name}")
                        markdown_content.append(f"\n![{scene_name}](图片生成失败)\n")
                else:
                    print(f"生成提示词失败: {scene_name}")
                    markdown_content.append(f"\n![{scene_name}](提示词生成失败)\n")
            
            # 添加尾注
            markdown_content.extend([
                "\n---\n",
                "**尾注：**\n"
            ])
            
            # 将内容写入文件
            with open(story_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(markdown_content))
            
            print(f"故事已保存到: {story_file}")
            return str(story_file)
            
        except Exception as e:
            print(f"处理故事时发生错误: {str(e)}")
            return None

    def format_story(self, story: Dict, image_links: List[str]) -> str:
        """
        将故事内容格式化为Markdown格式
        
        参数：
            story: 故事内容字典
            image_links: 图片链接列表
            
        返回：
            格式化后的Markdown文本
        """
        try:
            prompt = f"""请将以下故事内容格式化为Markdown格式的文档：

标题：{story["title"]}

角色：
{chr(10).join(story["characters"])}

段落：
{chr(10).join(story["paragraphs"])}

图片链接：
{chr(10).join(image_links)}

要求：
1. 使用Markdown语法
2. 标题使用一级标题
3. 在适当位置插入图片
4. 段落之间要有适当的空行
5. 保持整体排版美观"""
        
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
            return response.choices[0].message.content

        except Exception as e:
            print(f"格式化故事时发生错误: {str(e)}")
            return None

    def save_formatted_story(self, formatted_story: str, output_dir: str, title: str):
        """
        保存格式化后的故事到文件
        
        参数：
            formatted_story: markdown格式的故事文本
            output_dir: 输出目录
            title: 故事标题
        """
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 使用时间戳创建唯一的文件名
        filename = f"{title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(output_dir, filename)
        
        # 保存故事到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(formatted_story)
