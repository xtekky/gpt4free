# Reasoning Field Standardization

## Issue
DeepSeek uses `"reasoning_content"` field while OpenAI uses `"reasoning"` field in their chat completion streaming responses. This inconsistency caused confusion about what field name to use in the g4f Interference API.

## Decision
**Standardized on OpenAI's `"reasoning"` field format for API output while maintaining input compatibility.**

## Rationale
1. **OpenAI Compatibility**: OpenAI is the de facto standard for chat completion APIs
2. **Ecosystem Compatibility**: Most tools and libraries expect OpenAI format
3. **Consistency**: Provides a unified output format regardless of the underlying provider
4. **Backward Compatibility**: Input parsing continues to accept both formats

## Implementation

### Input Format Support (Unchanged)
The system continues to accept both input formats in `OpenaiTemplate.py`:
```python
reasoning_content = choice.get("delta", {}).get("reasoning_content", choice.get("delta", {}).get("reasoning"))
```

### Output Format Standardization (Changed)
- **Streaming Delta**: Uses `reasoning` field (OpenAI format)
- **Non-streaming Message**: Uses `reasoning` field (OpenAI format)  
- **API Responses**: Should use standard OpenAI streaming format

### Example Output Formats

#### Streaming Response (OpenAI Compatible)
```json
{
  "id": "chatcmpl-example",
  "object": "chat.completion.chunk",
  "choices": [{
    "index": 0,
    "delta": {
      "role": "assistant",
      "reasoning": "I need to think about this step by step..."
    },
    "finish_reason": null
  }]
}
```

#### Non-streaming Response
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Here's my answer",
      "reasoning": "My reasoning process was..."
    }
  }]
}
```

## Files Changed
- `g4f/client/stubs.py`: Updated to use `reasoning` field instead of `reasoning_content`

## Testing
- Added comprehensive tests for format standardization
- Verified input compatibility with both OpenAI and DeepSeek formats
- Confirmed no regressions in existing functionality