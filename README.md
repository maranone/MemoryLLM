# LLM Memory System with Ollama

A conversational AI system using Ollama with hybrid memory management that combines:
- Sliding window context (last 8-10 turns)
- Long-term vector memory storage (ChromaDB)
- Asynchronous memory updates
- Session persistence

## Features

✨ **Hybrid Memory Architecture**
- Maintains recent conversation context for natural flow
- Stores long-term memories in vector database
- Semantic search for relevant memory retrieval

🚀 **Asynchronous Processing**
- Non-blocking memory extraction and storage
- Smooth conversation experience
- Background memory updates

💾 **Session Management**
- Save and resume conversations
- Persistent memory across sessions
- Export conversation history

🎯 **Smart Memory Extraction**
- Automatic fact and preference detection
- Entity recognition
- Conversation summarization

## Prerequisites

1. **Python 3.10+**
2. **Ollama** - Install from [ollama.ai](https://ollama.ai)
3. **Gemma3:4b-it-qat model** - Pull with: `ollama pull gemma3:4b-it-qat`

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure Ollama is running:
```bash
ollama serve
```

4. Verify the model is available:
```bash
ollama list
```

## Usage

### Start the Chat System

```bash
python main.py
```

### Commands

- `/new` - Start a new conversation
- `/continue` - Load a previous session
- `/memories` - View stored memories
- `/save` - Save current session
- `/clear` - Clear context window (keeps long-term memory)
- `/exit` - Save and quit

### Example Conversation

```
You: Hi! My name is Alex and I'm a Python developer from Seattle.
Assistant: Hello Alex! Nice to meet you! It's great to connect with a fellow Python developer...

You: What's my name?
Assistant: Your name is Alex! You mentioned you're a Python developer from Seattle.

# System automatically extracts and stores:
# - FACT: User's name is Alex
# - FACT: User is a Python developer
# - FACT: User is from Seattle
```

The system will remember these facts even in new sessions!

## Configuration

Edit `config.py` to customize:

```python
# Model settings
OLLAMA_MODEL = "gemma3:4b-it-qat"
OLLAMA_HOST = "http://localhost:11434"

# Context window size
CONTEXT_WINDOW_SIZE = 10  # Recent turns to keep

# Memory retrieval
MEMORY_RETRIEVAL_COUNT = 5  # Top memories to retrieve

# Memory extraction frequency
EXTRACTION_FREQUENCY = 1  # Extract after every N turns
```

## Project Structure

```
llm-memory/
├── main.py                    # Entry point and chat loop
├── config.py                  # Configuration settings
├── requirements.txt           # Dependencies
├── memory/
│   ├── vector_store.py       # ChromaDB interface
│   ├── extractor.py          # Memory extraction logic
│   └── retriever.py          # Memory retrieval and formatting
├── conversation/
│   ├── ollama_client.py      # Ollama API wrapper
│   └── context_manager.py    # Sliding window management
├── sessions/                  # Saved conversation sessions
└── chroma_db/                # Vector database storage
```

## How It Works

### 1. Conversation Flow

1. User sends a message
2. System retrieves relevant memories from vector DB
3. Builds prompt with: system prompt + memories + recent context
4. Streams response from Ollama
5. Adds turn to context window (sliding window of last 10 turns)
6. Asynchronously extracts and stores new memories

### 2. Memory Types

- **Facts**: Personal information (name, location, occupation)
- **Preferences**: Likes, dislikes, favorites
- **Entities**: Names, places, dates mentioned
- **Summaries**: Conversation topic summaries

### 3. Memory Extraction

**Pattern-based extraction:**
- "I am/I'm [something]"
- "My name is [name]"
- "I like/love/prefer [thing]"
- "I live in [place]"

**LLM-based extraction:**
- Uses lightweight prompting to extract facts/preferences
- Runs asynchronously to avoid blocking conversation

### 4. Memory Retrieval

- Semantic similarity search using embeddings
- Top 5 most relevant memories retrieved per query
- Formatted and injected into system prompt
- Maintains conversation context across sessions

## Testing Scenarios

### Multi-turn Conversation
```
You: I prefer async code
Assistant: [responds]
You: Can you make that async?
Assistant: [understands "that" refers to previous code]
```

### Cross-session Memory
```
Session 1:
You: I love Python and FastAPI
[exit and restart]

Session 2:
You: What frameworks do I like?
Assistant: You mentioned you love FastAPI!
```

### Long Conversation Performance
The system maintains consistent performance even with 50+ turns by:
- Keeping only last 10 turns in active context
- Storing older information in vector DB
- Retrieving only relevant memories on-demand

## Troubleshooting

### Ollama Connection Error
```bash
# Start Ollama server
ollama serve

# Check if running
curl http://localhost:11434/api/tags
```

### Model Not Found
```bash
# Pull the model
ollama pull gemma3:4b-it-qat

# Verify
ollama list
```

### ChromaDB Issues
```bash
# Delete and reinitialize
rm -rf chroma_db/
# Restart the application
```

## Advanced Usage

### Using Different Models

Edit `config.py`:
```python
OLLAMA_MODEL = "llama3.2"  # or any other Ollama model
```

### Adjusting Context Window

Larger window = more recent context, more tokens:
```python
CONTEXT_WINDOW_SIZE = 15  # Increase from default 10
```

### Memory Management

View memories:
```bash
# In chat
/memories
```

Clear conversation memories:
```python
# In Python
vector_store.clear_conversation_memories(conversation_id)
```

## Dependencies

- **chromadb** - Vector database for memory storage
- **sentence-transformers** - Embedding generation
- **ollama** - Ollama Python client
- **rich** - Beautiful terminal UI
- **aiofiles** - Async file operations

## Performance Considerations

- **Memory extraction**: Runs asynchronously, doesn't block conversation
- **Context window**: Fixed size prevents token bloat
- **Vector search**: O(log n) retrieval time
- **Session files**: JSON format, easily portable

## Future Enhancements

- [ ] Memory importance scoring
- [ ] Memory decay (older memories fade)
- [ ] Multi-user support
- [ ] Web UI interface
- [ ] Conversation branching
- [ ] Memory conflict resolution
- [ ] Export to various formats

## License

MIT License - Feel free to use and modify!

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Built with [Ollama](https://ollama.ai)
- Uses [ChromaDB](https://www.trychroma.com/) for vector storage
- Inspired by human memory systems