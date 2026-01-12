import json
import os
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for API requests"""
    base_url: str = "https://yupp.ai"
    api_endpoint: str = "/api/trpc/model.getModelInfoList,scribble.getScribbleByLabel"
    timeout: int = 30
    fallback_file: str = "models.json"
    output_file: str = "model.json"


class YuppAPIClient:
    """Yupp API client for fetching model data"""
    
    def __init__(self, config: ModelConfig = None, api_key: str = None, session=None):
        self.config = config or ModelConfig()
        self.api_key = api_key
        if session is None:
            self.session = requests.Session()
            self._setup_session()
        else:
            self.session = session
        self._set_cookies()

    def _setup_session(self) -> None:
        """Setup session with headers and cookies"""
        self.session.headers.update(self._get_headers())
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": f"{self.config.base_url}/",
            "Origin": self.config.base_url,
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }
    
    def _set_cookies(self) -> None:
        """Set cookies from environment variable"""
        token = self._get_session_token()
        if token:
            self.session.cookies.set("__Secure-yupp.session-token", token)
    
    def _get_session_token(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        """Get session token from environment variable"""
        env_tokens = os.getenv("YUPP_TOKENS")
        if not env_tokens:
            return None
        
        try:
            tokens = [t.strip() for t in env_tokens.split(",") if t.strip()]
            return tokens[0] if tokens else None
        except Exception as e:
            print(f"Warning: Failed to parse YUPP_TOKENS: {e}")
            return None
    
    def _build_api_url(self) -> str:
        """Build the complete API URL"""
        params = "batch=1&input=%7B%220%22%3A%7B%22json%22%3Anull%2C%22meta%22%3A%7B%22values%22%3A%5B%22undefined%22%5D%7D%7D%2C%221%22%3A%7B%22json%22%3A%7B%22label%22%3A%22homepage_banner%22%7D%7D%7D"
        return f"{self.config.base_url}{self.config.api_endpoint}?{params}"
    
    def fetch_models(self) -> Optional[List[Dict[str, Any]]]:
        """Fetch model data from API"""
        url = self._build_api_url()
        
        try:
            print(f"Fetching data from: {url}")
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            
            data = response.json()
            print("Successfully fetched and parsed model data")
            
            # Extract model list from response structure
            if data and isinstance(data, list) and len(data) > 0:
                return data[0]["result"]["data"]["json"]
            else:
                print("Unexpected response format")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
        except (ValueError, json.JSONDecodeError) as e:
            print(f"JSON parsing failed: {e}")
            return None
        except KeyError as e:
            print(f"Data structure error - missing key: {e}")
            return None


class ModelProcessor:
    """Process and filter model data"""
    
    SUPPORTED_FAMILIES = {
        "GPT", "Claude", "Gemini", "Qwen", "DeepSeek", "Perplexity", "Kimi"
    }
    
    TAG_MAPPING = {
        "isPro": "☀️",
        "isMax": "🔥",
        "isNew": "🆕",
        "isLive": "🎤",
        "isAgent": "🤖",
        "isFast": "🚀",
        "isReasoning": "🧠",
        "isImageGeneration": "🎨",
    }
    
    @classmethod
    def generate_tags(cls, item: Dict[str, Any]) -> List[str]:
        """Generate tags for model display"""
        tags = []
        
        # Add emoji tags based on boolean flags
        for key, emoji in cls.TAG_MAPPING.items():
            if item.get(key, False):
                tags.append(emoji)
        
        # Add attachment tag if supported
        if item.get("supportedAttachmentMimeTypes"):
            tags.append("📎")
        
        return tags
    
    @classmethod
    def should_include_model(cls, item: Dict[str, Any]) -> bool:
        """Check if model should be included in output"""
        family = item.get("family")
        
        # Include if in supported families or has special features
        return (
            family in cls.SUPPORTED_FAMILIES or
            item.get("isImageGeneration") or
            item.get("isAgent") or
            item.get("isLive")
        )
    
    @classmethod
    def process_model_item(cls, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual model item"""
        tags = cls.generate_tags(item)
        label = item.get("label", "")
        
        # Add tags to label if present
        if tags:
            label += "\n" + " | ".join(tags)
        
        return {
            "id": item.get("id"),
            "name": item.get("name"),
            "label": label,
            "shortLabel": item.get("shortLabel"),
            "publisher": item.get("publisher"),
            "family": item.get("family"),
            "isPro": item.get("isPro", False),
            "isInternal": item.get("isInternal", False),
            "isMax": item.get("isMax", False),
            "isLive": item.get("isLive", False),
            "isNew": item.get("isNew", False),
            "isImageGeneration": item.get("isImageGeneration", False),
            "isAgent": item.get("isAgent", False),
            "isReasoning": item.get("isReasoning", False),
            "isFast": item.get("isFast", False),
        }
    
    @classmethod
    def filter_and_process(cls, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and process model data"""
        return [
            cls.process_model_item(item)
            for item in data
            if cls.should_include_model(item)
        ]


class DataManager:
    """Handle data loading and saving operations"""
    
    @staticmethod
    def load_fallback_data(filename: str) -> List[Dict[str, Any]]:
        """Load fallback data from local file"""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Fallback file not found: {filename}")
            return []
        except json.JSONDecodeError as e:
            print(f"Failed to parse fallback file: {e}")
            return []
    
    @staticmethod
    def save_data(data: List[Dict[str, Any]], filename: str) -> bool:
        """Save data to JSON file"""
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", 
                      exist_ok=True)
            
            # Create file if it doesn't exist
            if not os.path.exists(filename):
                open(filename, "a", encoding="utf-8").close()
            
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            print(f"Successfully saved {len(data)} models to {filename}")
            return True
            
        except Exception as e:
            print(f"Failed to save data: {e}")
            return False


class YuppModelManager:
    """Main manager class for Yupp model operations"""
    
    def __init__(self, config: ModelConfig = None, api_key: str = None, session=None):
        self.config = config or ModelConfig()
        self.client = YuppAPIClient(config, api_key, session)
        self.processor = ModelProcessor()
        self.data_manager = DataManager()
    
    def has_valid_token(self) -> bool:
        """Check if valid token is available"""
        return self.client._get_session_token() is not None
    
    def fetch_and_save_models(self, output_file: str = None) -> bool:
        """Main method to fetch and save model data"""
        output_file = output_file or self.config.output_file
        
        print("=== Yupp Model Data Fetcher ===")
        
        if not self.has_valid_token():
            print("Warning: YUPP_TOKENS environment variable not set")
            return False
        
        # Try to fetch from API
        data = self.client.fetch_models()
        
        # Fallback to local data if API fails
        if not data:
            print("API request failed, trying fallback data...")
            data = self.data_manager.load_fallback_data(self.config.fallback_file)
        
        if not data:
            print("No model data available")
            return False
        
        print(f"Processing {len(data)} models...")
        processed_models = self.processor.filter_and_process(data)
        
        return self.data_manager.save_data(processed_models, output_file)
    
    def run_interactive(self) -> bool:
        """Run in interactive mode (for CLI use)"""
        
        print("=== Yupp Model Data Tool ===")
        
        if not self.has_valid_token():
            print("Error: YUPP_TOKENS environment variable not set")
            print("Please set YUPP_TOKENS environment variable, e.g.:")
            print("export YUPP_TOKENS='your_token_here'")
            return False
        
        return self.fetch_and_save_models()


def main():
    """Main entry point"""
    manager = YuppModelManager()
    success = manager.run_interactive()
    
    if success:
        print("Operation completed successfully")
    else:
        print("Operation failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
