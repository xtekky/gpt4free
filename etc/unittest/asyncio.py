from .include import DEFAULT_MESSAGES
import asyncio
try:
    import nest_asyncio
    has_nest_asyncio = True
except:
    has_nest_asyncio = False 
import unittest
import g4f
from g4f import ChatCompletion
from .mocks import ProviderMock, AsyncProviderMock, AsyncGeneratorProviderMock
        
class TestChatCompletion(unittest.TestCase):
    
    async def run_exception(self):
        return ChatCompletion.create(g4f.models.default, DEFAULT_MESSAGES, AsyncProviderMock)
        
    def test_exception(self):
        self.assertRaises(g4f.errors.NestAsyncioError, asyncio.run, self.run_exception())

    def test_create(self):
        result = ChatCompletion.create(g4f.models.default, DEFAULT_MESSAGES, AsyncProviderMock)
        self.assertEqual("Mock",result)
        
    def test_create_generator(self):
        result = ChatCompletion.create(g4f.models.default, DEFAULT_MESSAGES, AsyncGeneratorProviderMock)
        self.assertEqual("Mock",result)

class TestChatCompletionAsync(unittest.IsolatedAsyncioTestCase):
    
    async def test_base(self):
        result = await ChatCompletion.create_async(g4f.models.default, DEFAULT_MESSAGES, ProviderMock)
        self.assertEqual("Mock",result)
        
    async def test_async(self):
        result = await ChatCompletion.create_async(g4f.models.default, DEFAULT_MESSAGES, AsyncProviderMock)
        self.assertEqual("Mock",result)
        
    async def test_create_generator(self):
        result = await ChatCompletion.create_async(g4f.models.default, DEFAULT_MESSAGES, AsyncGeneratorProviderMock)
        self.assertEqual("Mock",result)
        
class TestChatCompletionNestAsync(unittest.IsolatedAsyncioTestCase):
        
    def setUp(self) -> None:
        if not has_nest_asyncio:
            self.skipTest('"nest_asyncio" not installed')
        nest_asyncio.apply()
        
    async def test_create(self):
        result = await ChatCompletion.create_async(g4f.models.default, DEFAULT_MESSAGES, ProviderMock)
        self.assertEqual("Mock",result)
        
    async def test_nested(self):
        result = ChatCompletion.create(g4f.models.default, DEFAULT_MESSAGES, AsyncProviderMock)
        self.assertEqual("Mock",result)
        
    async def test_nested_generator(self):
        result = ChatCompletion.create(g4f.models.default, DEFAULT_MESSAGES, AsyncGeneratorProviderMock)
        self.assertEqual("Mock",result)

if __name__ == '__main__':
    unittest.main()