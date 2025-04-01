import os
import asyncio
import aiohttp
from g4f.client import Client

class FluxImageGenerator:
    """
    Flux图像生成器类
    负责调用Flux API生成图像
    """
    
    def __init__(self):
        pass
        
    async def _generate_image_async(self, 
                                  positive_prompt: str, 
                                  negative_prompt: str,
                                  output_path: str) -> bool:
        """
        异步调用Flux API生成图像
        
        参数：
            positive_prompt: 正向提示词
            negative_prompt: 负向提示词
            output_path: 图像保存路径
            
        返回：
            bool: 是否成功生成图像
        """
        try:
            print(f"\n开始生成图像...")
            print(f"提示词: {positive_prompt}")
            print(f"负向提示词: {negative_prompt}")
            
            
            # 调用Flux API生成图像
            
            client = Client()
            result = await client.images.async_generate(
                model="flux",
                prompt=f"{positive_prompt}",
                response_format="url"
            )

            print(f"\nAPI响应: {result}")

            # 检查API响应结构
            if result.data[0].url:
                image_url = result.data[0].url
                print(f"获取到图像URL: {image_url}")

                # 下载图像
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_url) as img_response:
                        print("+++++++", img_response.status, output_path)
                        if img_response.status == 200:
                            # 确保输出目录存在
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            with open(output_path, 'wb') as f:
                                f.write(await img_response.read())
                            print(f"图像已保存到: {output_path}")
                            return True
                        else:
                            print(f"下载图像失败: HTTP {img_response.status}")
            else:
                print(f"API响应格式不正确: {result}")
            
            return False
            
        except Exception as e:
            print(f"生成图像时发生错误: {str(e)}")
            if hasattr(e, 'response'):
                print(f"API错误响应: {e.response.text if hasattr(e.response, 'text') else e.response}")
            return False
            
    def generate_image(self, 
                      positive_prompt: str, 
                      negative_prompt: str,
                      output_path: str) -> bool:
        """
        同步方式调用Flux API生成图像
        
        参数：
            positive_prompt: 正向提示词
            negative_prompt: 负向提示词
            output_path: 图像保存路径
            
        返回：
            bool: 是否成功生成图像
        """
        try:
            # 获取或创建事件循环
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # 运行异步函数
            result = loop.run_until_complete(
                self._generate_image_async(
                    positive_prompt,
                    negative_prompt,
                    output_path
                )
            )
            return result
            
        except Exception as e:
            print(f"生成图像时发生错误: {str(e)}")
            return False