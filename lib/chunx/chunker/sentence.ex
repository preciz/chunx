defmodule Chunx.Chunker.Sentence do
  @moduledoc """
  Implements sentence based chunking strategy.

  Splits text into overlapping chunks based on sentences while
  respecting token limits.
  """

  alias Chunx.Chunk
  alias Chunx.Chunker.SentenceSplitter
  alias Chunx.SentenceChunk

  @behaviour Chunx.Chunker

  @type chunk_opts :: [
          chunk_size: pos_integer(),
          chunk_overlap: pos_integer(),
          min_sentences_per_chunk: pos_integer(),
          delimiters: list(String.t()),
          short_sentence_threshold: pos_integer()
        ]

  @default_opts [
    chunk_size: 512,
    chunk_overlap: 128,
    min_sentences_per_chunk: 1,
    delimiters: ~w(. ! ? \\n),
    short_sentence_threshold: 6
  ]

  @doc """
  Splits text into overlapping chunks using sentence boundaries.

  ## Options
    * `:chunk_size` - Maximum number of tokens per chunk (default: 512). The chunker will try to fit
      as many complete sentences as possible while staying under this limit. If a single sentence
      exceeds this limit, it will still be included as its own chunk.

    * `:chunk_overlap` - Number of tokens that should overlap between consecutive chunks (default: 128).
      This helps maintain context between chunks by including some sentences from the end of the previous
      chunk at the start of the next chunk. Must be less than chunk_size.

    * `:min_sentences_per_chunk` - Minimum number of sentences that must be included in each chunk
      (default: 1). This ensures chunks contain complete thoughts, even if including multiple sentences
      would exceed chunk_size.

    * `:delimiters` - List of sentence delimiters. Sentences will be split
      at these delimiters. (default: ["." "!" "?" "\\n"])

    * `:short_sentence_threshold` - Below this byte size a sentence is considered too short and will be
       concatenated with the next sentence. (default: 6)

  """
  @spec chunk(binary(), Tokenizers.Tokenizer.t(), chunk_opts()) ::
          {:ok, [SentenceChunk.t()]} | {:error, term()}
  def chunk(text, tokenizer, opts \\ []) when is_binary(text) do
    opts = Keyword.merge(@default_opts, opts)
    config = validate_config!(opts)

    if String.trim(text) == "" do
      {:ok, []}
    else
      chunks =
        text
        |> prepare_sentences(tokenizer, config)
        |> create_chunks(tokenizer, config)

      {:ok, chunks}
    end
  end

  defp validate_config!(opts) do
    chunk_size = Keyword.fetch!(opts, :chunk_size)
    chunk_overlap = Keyword.fetch!(opts, :chunk_overlap)
    min_sentences = Keyword.fetch!(opts, :min_sentences_per_chunk)
    delimiters = Keyword.fetch!(opts, :delimiters)
    short_sentence_threshold = Keyword.fetch!(opts, :short_sentence_threshold)

    validate_positive_integer!(chunk_size, "chunk_size must be positive")
    validate_overlap!(chunk_overlap, chunk_size)

    validate_positive_integer!(
      min_sentences,
      "min_sentences_per_chunk must be at least 1"
    )

    validate_delimiters!(delimiters)

    validate_positive_integer!(
      short_sentence_threshold,
      "short_sentence_threshold must be at least 1"
    )

    %{
      chunk_size: chunk_size,
      chunk_overlap: chunk_overlap,
      min_sentences_per_chunk: min_sentences,
      delimiters: delimiters,
      short_sentence_threshold: short_sentence_threshold
    }
  end

  defp validate_positive_integer!(value, _message) when is_integer(value) and value > 0,
    do: :ok

  defp validate_positive_integer!(_value, message), do: raise(ArgumentError, message)

  defp validate_overlap!(overlap, chunk_size)
       when is_integer(overlap) and overlap >= 0 and overlap < chunk_size,
       do: :ok

  defp validate_overlap!(_overlap, _chunk_size),
    do: raise(ArgumentError, "chunk_overlap must be less than chunk_size")

  defp validate_delimiters!(delimiters) when is_list(delimiters) and delimiters != [] do
    if Enum.all?(delimiters, &(is_binary(&1) and &1 != "")),
      do: :ok,
      else: raise(ArgumentError, "delimiters must contain at least one element")
  end

  defp validate_delimiters!(_delimiters),
    do: raise(ArgumentError, "delimiters must contain at least one element")

  defp split_sentences(text, config) do
    text
    |> SentenceSplitter.split(config.delimiters)
    |> combine_short_sentences([], config.short_sentence_threshold)
  end

  defp combine_short_sentences([], acc, _), do: Enum.reverse(acc)

  defp combine_short_sentences([sentence | rest], [], threshold)
       when byte_size(sentence) < threshold do
    combine_short_sentences(rest, [sentence], threshold)
  end

  defp combine_short_sentences([sentence | rest], [prev | remaining], threshold)
       when byte_size(sentence) < threshold do
    combine_short_sentences(rest, [prev <> sentence | remaining], threshold)
  end

  defp combine_short_sentences([sentence | rest], acc, threshold) do
    combine_short_sentences(rest, [sentence | acc], threshold)
  end

  defp prepare_sentences(text, tokenizer, config) do
    text
    |> split_sentences(config)
    |> convert_sentences_to_chunks(tokenizer, config)
  end

  defp convert_sentences_to_chunks(sentences, tokenizer, _config) do
    sentences
    |> Enum.reduce({0, []}, fn sentence, {pos, acc} ->
      {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, sentence)
      token_count = Tokenizers.Encoding.get_length(encoding)

      chunk = %Chunk{
        text: sentence,
        start_byte: pos,
        end_byte: pos + byte_size(sentence),
        token_count: token_count
      }

      {pos + byte_size(sentence), [chunk | acc]}
    end)
    |> elem(1)
    |> Enum.reverse()
  end

  defp create_chunks(sentences, tokenizer, config) do
    do_create_chunks(sentences, tokenizer, config, [], 0)
  end

  defp do_create_chunks(sentences, _tokenizer, _config, sentence_chunks, pos)
       when pos >= length(sentences) do
    Enum.reverse(sentence_chunks)
  end

  defp do_create_chunks(sentences, tokenizer, config, sentence_chunks, pos) do
    {chunk_sentences, split_idx} = split_at_chunk_boundary(sentences, pos, config)

    case create_sentence_chunk(chunk_sentences, tokenizer) do
      %SentenceChunk{} = sentence_chunk ->
        next_pos = find_overlap_start(chunk_sentences, split_idx, length(sentences), config)

        do_create_chunks(
          sentences,
          tokenizer,
          config,
          [sentence_chunk | sentence_chunks],
          next_pos
        )
    end
  end

  defp split_at_chunk_boundary(sentences, pos, config) do
    {chunk_sentences, _total} =
      sentences
      |> Enum.drop(pos)
      |> Enum.reduce_while({[], 0}, fn sentence, {acc, current_tokens} ->
        total_tokens = current_tokens + sentence.token_count

        if total_tokens <= config.chunk_size or length(acc) < config.min_sentences_per_chunk do
          {:cont, {[sentence | acc], total_tokens}}
        else
          {:halt, {acc, current_tokens}}
        end
      end)

    chunk_sentences = Enum.reverse(chunk_sentences)

    {chunk_sentences, pos + length(chunk_sentences)}
  end

  defp create_sentence_chunk(sentences, tokenizer) do
    text = Enum.map_join(sentences, "", & &1.text)
    {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, text)

    %SentenceChunk{
      text: text,
      start_byte: hd(sentences).start_byte,
      end_byte: List.last(sentences).end_byte,
      token_count: Tokenizers.Encoding.get_length(encoding),
      sentences: sentences
    }
  end

  defp find_overlap_start(chunk_sentences, split_idx, total_len, config) do
    if config.chunk_overlap > 0 and split_idx < total_len do
      calculate_sentence_overlap(chunk_sentences, split_idx, config.chunk_overlap)
    else
      split_idx
    end
  end

  defp calculate_sentence_overlap(chunk_sentences, split_idx, chunk_overlap) do
    {overlap_pos, _} =
      chunk_sentences
      |> Enum.reverse()
      |> Enum.reduce_while({split_idx, 0}, fn sentence, {current_idx, total_tokens} ->
        new_total = total_tokens + sentence.token_count

        if new_total > chunk_overlap do
          {:halt, {current_idx, new_total}}
        else
          {:cont, {current_idx - 1, new_total}}
        end
      end)

    overlap_pos
  end
end
