# Changelog

All notable changes to Chunx are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-08-21

### Added

- Added `Chunx.Chunker.Recursive`, which progressively splits text at
  paragraph, sentence, punctuation, whitespace, and token boundaries.
- Added custom tokenizer adapters through `Chunx.Tokenizer`.
- Added an example script covering the available non-semantic chunkers.
- Added opt-in real-model embedding integration tests, runnable with
  `mix test --include integration` or `mix test --only integration`.
- Added property tests for reconstruction, offsets, overlap, Unicode, tokenizer
  windows, and semantic metadata.

### Changed

- Standardized `token_count` across all chunkers to mean content tokens,
  excluding tokenizer entries without a byte span.
- Made tokenizer-derived chunk boundaries grapheme-safe.
- Reduced intermediate allocations and repeated traversal in the chunkers.
- Updated Nx, EXLA, Bumblebee, Scholar, StreamData, and related dependencies.

### Fixed

- Preserved exact byte offsets for repeated text and Unicode content.
- Prevented Sentence from stalling when overlap contains the entire
  previous chunk.
- Preserved Token's trailing overlap window while keeping indivisible
  graphemes intact.
- Propagated tokenizer failures instead of crashing on failed encodings.
- Rejected malformed tokenizer responses and offsets with tagged errors.
- Prevented Semantic from silently dropping sentences when an embedding
  function returns the wrong number of embeddings.
- Preserved all input text when grouping sentences and placing chunk boundaries.

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
