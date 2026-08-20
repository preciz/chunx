# Changelog

All notable changes to Chunx are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-08-21

### Added

- Added `Chunx.Chunker.Recursive`, which progressively splits text at
  paragraph, sentence, punctuation, whitespace, and token boundaries.
- Added a shared tokenizer boundary with support for native
  `Tokenizers.Tokenizer` values and custom tokenizer adapters.
- Added grapheme-safe handling for repeated, overlapping, and byte-level token
  offsets.
- Added an example script covering the available non-semantic chunkers.
- Added opt-in real-model embedding integration tests, runnable with
  `mix test --include integration` or `mix test --only integration`.
- Expanded property testing for reconstruction, offsets, overlap normalization,
  semantic metadata, sentence grouping, tokenizer windows, Unicode, and
  statistical invariants.

### Changed

- Standardized `token_count` across all chunkers to mean content tokens,
  excluding tokenizer entries without a byte span.
- Refactored Token, Word, Sentence, Recursive, and Semantic chunking paths to
  reduce intermediate collections and repeated traversal.
- Optimized native token counting and ordinary token-window packing while
  retaining the hardened Unicode fallback.
- Simplified semantic similarity averaging without changing the adjacent
  cosine-similarity model.
- Updated Nx, EXLA, Bumblebee, Scholar, StreamData, and related dependencies.

### Fixed

- Preserved exact byte offsets for repeated text and Unicode content.
- Prevented SentenceChunker from stalling when overlap contains the entire
  previous chunk.
- Preserved TokenChunker's trailing overlap window while keeping indivisible
  graphemes intact.
- Propagated tokenizer failures consistently instead of crashing on failed
  encodings.
- Rejected malformed tokenizer responses and offsets with tagged errors.
- Prevented SemanticChunker from silently dropping sentences when an embedding
  function returns the wrong number of embeddings.
- Corrected sentence grouping and chunk-boundary behavior to preserve all input
  text.

## [0.1.0] - 2026-03-04

### Added

- Initial Hex.pm release.
- Added Token, Word, Sentence, and Semantic chunking strategies.
- Added configurable chunk sizes, overlap, sentence grouping, and semantic
  similarity thresholds.
- Added byte-offset and token-count metadata through `Chunx.Chunk` and
  `Chunx.SentenceChunk`.
- Added caller-provided Nx embedding support for semantic chunking.

[Unreleased]: https://github.com/preciz/chunx/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/preciz/chunx/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/preciz/chunx/tree/v0.1.0
