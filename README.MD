RAG on Obsidian – Full Project Roadmap
Phase 1 — Core Local RAG Pipeline

    Local Embedding Model

        Use sentence-transformers (e.g., BAAI/bge-base-en) on RTX 4050.

        Enable GPU (cuda) with batch embedding.

    Custom Incremental Loader for Obsidian

        Scan vault for .md files.

        Store metadata:

            file_path

            SHA256 hash or last modified timestamp

            chunk_ids

        Only embed new/changed files.

        Remove vectors for deleted files.

    Chunking

        Use MarkdownTextSplitter with tuned:

            chunk_size

            chunk_overlap

    Embedding & Storage

        Batch embeddings locally.

        Store in ChromaDB with metadata:

            file_path

            last_modified

            chunk_index

Phase 2 — Incremental Sync

    File Watcher

        Use watchdog to detect:

            Create

            Update

            Delete events

        Apply only necessary embedding/deletion.

    Scheduled Backup Scan

        Full vault re-check daily for safety.

Phase 3 — Backend API (FastAPI)

    Endpoints

        POST /query — Vault-wide RAG.

        POST /query-doc — Single-document RAG (pass file path).

        POST /sync — Trigger embedding sync.

        GET /files — List indexed files & last modified.

        POST /settings — Vault path + Groq API key.

        POST /voice-query — Accept audio, transcribe with Whisper, run RAG.

    RAG Output Format

        Answer text.

        List of sources:

            file_path

            Chunk preview.

        Style like Perplexity — clickable to open in Obsidian or preview.

Phase 4 — Voice-to-Text (Self-Hosted Whisper)

    Local Whisper Inference

        Use faster-whisper or whisper.cpp.

        Accept .wav or .mp3.

        Stream transcription to /voice-query.

Phase 5 — Frontend (Tauri + React)

    Chat UI

        Streaming Groq answers.

        Sources panel with clickable references.

        “Open in Obsidian” button.

    Voice Input

        Microphone button → record → send to /voice-query.

    Mode Toggle

        Vault mode — multi-doc retrieval.

        Single-doc mode — select file before query.

    Settings Page

        Vault path.

        Groq API key.

        Retrieval k value.

    Sync Status

        Show last sync time & number of files indexed.

Phase 6 — Optional Performance Boost

(Only if speed is slow)
15. GPU Embedding Optimizations
- Batch size tuning.
- Mixed precision (fp16).

    Async Background Processing

        Non-blocking sync for responsive UI.

✅ Final Product Features

    Vault-wide or single-doc RAG.

    Perplexity-style source links.

    Voice input via Whisper.

    Real-time note updates.

    Fully local with ChromaDB + GPU embeddings.

    Optimized for instant UX.