import os
import time
from pathlib import Path
import sys
from g4stories.story_config import StoryConfig
from g4stories.flux_prompt_generator import FluxPromptGenerator
from g4stories.flux_image_generator import FluxImageGenerator
from g4stories.story_generator import StoryGenerator
from g4stories.story_formatter import StoryFormatter

def main():
    """
    主函数：批量读取test.md文件并生成故事
    """
    # 获取API密钥和基础URL

    fal_key = os.getenv("FAL_KEY")
    
    # 获取输入文件路径
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "test.md"
    
    # 创建输出目录
    stories_dir = Path("generated_stories")
    images_dir = Path("generated_images")
    stories_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    
    # 创建故事配置
    config = StoryConfig(
        language="中文",
        target_age="5-8岁",
        words_per_paragraph=100,
        paragraph_count=12  # 增加到12段
    )
    
    # 初始化各个组件
    story_generator = StoryGenerator()
    prompt_generator = FluxPromptGenerator()
    image_generator = FluxImageGenerator()
    story_formatter = StoryFormatter()
    
    # 读取输入文件
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            themes = [line.strip() for line in f if line.strip()]
        
        print(f"找到 {len(themes)} 个故事主题")
        
        # 为每个主题生成故事
        for i, theme in enumerate(themes, 1):
            print(f"\n正在生成第 {i} 个故事: {theme}")
            
            # 生成故事
            story = story_generator.generate_story(
                theme=theme,
                config=config,
                additional_requirements="故事要富有教育意义，适合儿童阅读"
            )
            
            if story:
                image_links = []
                # 为每个段落生成配图
                if isinstance(story.get('paragraphs', []), list):
                    for j, paragraph in enumerate(story['paragraphs']):
                        # 获取段落内容和场景描述
                        if isinstance(paragraph, dict):
                            content = paragraph.get('paragraph', '')
                            scene = paragraph.get('scene', '')
                        else:
                            content = paragraph
                            scene = content  # 如果没有场景描述，使用段落内容
                        
                        # 生成图片提示词
                        prompts = prompt_generator.generate_prompts(
                            title=story['title'],
                            scene=scene,
                            main_character=story.get('main_character', '')
                        )
                        
                        if prompts:
                            # 生成图片
                            image_path = images_dir / f"{theme}_scene_{j+1}.png"
                            success = image_generator.generate_image(
                                positive_prompt=prompts['positive_prompt'],
                                negative_prompt=prompts['negative_prompt'],
                                output_path=str(image_path)
                            )
                            if success:
                                # 使用相对路径保存图片链接
                                image_links.append(f"../generated_images/{image_path.name}")
                
                # 格式化故事
                formatted_story = story_formatter.format_story(story, image_links)
                if formatted_story:
                    # 保存格式化后的故事
                    story_formatter.save_formatted_story(
                        formatted_story=formatted_story,
                        output_dir=str(stories_dir),
                        title=theme
                    )
                    print(f"故事已保存到: {stories_dir}/{theme}.md")
                else:
                    print(f"格式化故事 '{theme}' 失败")
            else:
                print(f"生成故事 '{theme}' 失败")
            
            # 添加延时以避免API限制
            time.sleep(2)
        
        print("\n所有故事生成完成！")
        print(f"故事文件保存在 {stories_dir} 目录下")
        print(f"图片文件保存在 {images_dir} 目录下")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()