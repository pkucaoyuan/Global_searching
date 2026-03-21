# Transcribe Lecture

Transcribe course audio recordings to text using Azure OpenAI Whisper API.

## Usage

```
/transcribe-lecture <audio_file> [--output <path>] [--update-rag]
```

**Parameters:**
- `audio_file`: Path to audio file (m4a, mp3, wav)
- `--output`: Output file path (default: same name with .txt extension in records/)
- `--update-rag`: Update RAG knowledge base after transcription (default: true)

**Examples:**
```
/transcribe-lecture records/Lecture\ 2.m4a
/transcribe-lecture meetings/office_hours.mp3 --output records/office_hours_02_07.txt
/transcribe-lecture records/Lecture\ 3.m4a --update-rag
```

---

## Instructions for Claude

When the user invokes `/transcribe-lecture`, follow these steps:

### Step 1: Validate Input

1. Check if the audio file exists
2. Check file size (>25MB requires chunking)
3. Check format (m4a needs conversion to mp3)

### Step 2: Convert Format (if needed)

If the file is `.m4a`, convert to `.mp3` using ffmpeg:

```bash
ffmpeg -i "input.m4a" -codec:a libmp3lame -b:a 128k "output.mp3"
```

Save temporary mp3 to `tmp/` directory.

### Step 3: Handle Large Files

If file is >20MB after conversion, split into chunks:

```bash
# Get duration in seconds
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "input.mp3"

# Split into 15-minute segments (~20MB at 128kbps)
ffmpeg -i "input.mp3" -f segment -segment_time 900 -c copy "tmp/chunk_%03d.mp3"
```

### Step 4: Transcribe

Run the transcription script for each chunk:

```bash
python scripts/transcribe_azure_audio.py "chunk.mp3" "chunk.txt"
```

**Environment Requirements:**
- `AZURE_OPENAI_API_KEY` must be set
- API endpoint: Azure OpenAI gpt-4o-mini-audio-preview

### Step 5: Merge Transcripts

If multiple chunks, merge with timestamps:

```
=== Segment 1 (0:00 - 15:00) ===
[transcript 1]

=== Segment 2 (15:00 - 30:00) ===
[transcript 2]

...
```

### Step 6: Save Output

1. Save merged transcript to `records/Lecture X.txt`
2. Clean up temporary files in `tmp/`

### Step 7: Update RAG (if --update-rag)

Add entry to `.claude/knowledge_base/knowledge_sources/lecture_transcripts.md`:

```markdown
### Lecture X - [Topic] ([Date])
- **File**: `records/Lecture X.txt` (XXX chars, ~YY min)
- **Audio**: `records/Lecture X.m4a` (ZZ MB)
- **Transcription**: Azure Whisper API
- **Topics Covered**:
  - Topic 1
  - Topic 2
```

Also update `records_index.md` in both:
- `.claude/knowledge_base/knowledge_sources/records_index.md`
- `/Users/ruicheng/Documents/GitHub/micromaster/.claude/knowledge_base/knowledge_sources/records_index.md`

### Step 8: Rebuild Embedding Index

After transcription, rebuild the semantic search index:

```bash
cd /Users/ruicheng/Documents/GitHub/micromaster
python scripts/rag/build_nonlinear_opt_rag.py --force
```

This updates `micromaster/data/nonlinear_opt_rag_index.pkl` with the new lecture content.

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    /transcribe-lecture                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Check Format   │
                    │  m4a/mp3/wav?   │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
      ┌───────────────┐           ┌─────────────────┐
      │ m4a → Convert │           │  mp3/wav ready  │
      │ ffmpeg → mp3  │           │                 │
      └───────┬───────┘           └────────┬────────┘
              │                            │
              └──────────────┬─────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Check Size     │
                    │  >20MB?         │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
      ┌───────────────┐           ┌─────────────────┐
      │ Split chunks  │           │  Single file    │
      │ 15min each    │           │                 │
      └───────┬───────┘           └────────┬────────┘
              │                            │
              ▼                            ▼
      ┌───────────────┐           ┌─────────────────┐
      │ Transcribe    │           │  Transcribe     │
      │ each chunk    │           │  single file    │
      └───────┬───────┘           └────────┬────────┘
              │                            │
              ▼                            │
      ┌───────────────┐                    │
      │ Merge chunks  │                    │
      └───────┬───────┘                    │
              │                            │
              └──────────────┬─────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Save to        │
                    │  records/       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Update RAG     │
                    │  (both repos)   │
                    └─────────────────┘
```

---

## API Configuration

**Azure OpenAI Endpoint:**
```
https://85409-mk9jw7uo-eastus2.openai.azure.com/openai/deployments/gpt-4o-mini-audio-preview/chat/completions
```

**API Version:** `2025-01-01-preview`

**Supported Formats:** mp3, wav (NOT m4a directly)

**File Size Limits:**
- Single request: ~25MB max
- Recommended chunk: 15 min (~20MB at 128kbps)

**Token Limits:**
- max_tokens: 16000 (sufficient for ~15 min audio)

---

## Error Handling

| Error | Solution |
|-------|----------|
| `AZURE_OPENAI_API_KEY not set` | Run: `export AZURE_OPENAI_API_KEY='...'` |
| `Unsupported format: .m4a` | Convert: `ffmpeg -i input.m4a -codec:a libmp3lame output.mp3` |
| `File too large` | Split into chunks |
| `API timeout` | Increase timeout or use smaller chunks |
| `HTTP 429 Rate limit` | Wait and retry with exponential backoff |

---

## Output Format

### Transcript File Structure

```
MIT 6.7220/15.084 Nonlinear Optimization
Lecture [N]: [Topic]
Date: [Date]
Duration: [X] minutes
Transcribed: [Timestamp]

================================================================================

[Verbatim transcript content...]

================================================================================

End of Transcript
```

---

## Subagent Configuration

This skill can use a `transcribe-lecture` subagent:

**Agent Type**: `Bash`
**Tools Available**: Bash (for ffmpeg, python)
**Capabilities**:
- Convert audio formats
- Split long files
- Run transcription script
- Clean up temp files

**Pre-flight Checks:**
```bash
# Verify ffmpeg installed
which ffmpeg

# Verify API key set
echo $AZURE_OPENAI_API_KEY | head -c 10

# Verify Python script exists
ls scripts/transcribe_azure_audio.py
```
