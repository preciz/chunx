defmodule Chunx.Chunker.Sentence do
  @moduledoc """
  Splits text at sentence boundaries, with optional whole-sentence overlap.
  """

  alias Chunx.Chunk
  alias Chunx.Chunker.SentenceSplitter
  alias Chunx.SentenceChunk
  alias Chunx.Tokenizer

  @behaviour Chunx.Chunker

  @type chunk_opts :: [
          chunk_size: pos_integer(),
          chunk_overlap: non_neg_integer(),
          min_sentences_per_chunk: pos_integer(),
          delimiters: nonempty_list(String.t()),
          short_sentence_threshold: pos_integer()
        ]

  @default_opts [
    chunk_size: 512,
    chunk_overlap: 128,
    min_sentences_per_chunk: 1,
    delimiters: [".", "!", "?", "\n"],
    short_sentence_threshold: 6
  ]

  @doc """
  Splits text into overlapping chunks using sentence boundaries.

  ## Options
    * `:chunk_size` - Target maximum content-token count (default: 512). Whole
      sentences and `:min_sentences_per_chunk` take precedence over this target.

    * `:chunk_overlap` - Token budget for whole sentences repeated between
      consecutive chunks (default: 128). Must be non-negative and less than
      `:chunk_size`.

    * `:min_sentences_per_chunk` - Minimum sentence count before applying the
      size target (default: 1).

    * `:delimiters` - Strings that end sentences (default:
      `[".", "!", "?", "\\n"]`).

    * `:short_sentence_threshold` - Sentences shorter than this byte count are
      joined to an adjacent sentence (default: 6).
  """
  @spec chunk(binary(), Tokenizer.t(), chunk_opts()) ::
          {:ok, [SentenceChunk.t()]} | {:error, term()}
  def chunk(text, tokenizer, opts \\ []) when is_binary(text) do
    opts = Keyword.merge(@default_opts, opts)
    config = validate_config!(opts)

    if String.trim(text) == "" do
      {:ok, []}
    else
      with {:ok, sentences} <- prepare_sentences(text, tokenizer, config) do
        create_nonempty_chunks(sentences, tokenizer, config)
      end
    end
  end

  defp create_nonempty_chunks(sentences, tokenizer, config) do
    if Enum.all?(sentences, &(&1.token_count == 0)),
      do: {:ok, []},
      else: create_chunks(sentences, tokenizer, config)
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
      else: raise(ArgumentError, "delimiters must contain non-empty strings")
  end

  defp validate_delimiters!(_delimiters),
    do: raise(ArgumentError, "delimiters must contain non-empty strings")

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
    |> convert_sentences_to_chunks(tokenizer)
  end

  defp convert_sentences_to_chunks(sentences, tokenizer) do
    result =
      Enum.reduce_while(sentences, {0, []}, &add_sentence(&1, &2, tokenizer))

    case result do
      {:error, _reason} = error -> error
      {_end_byte, chunks} -> {:ok, Enum.reverse(chunks)}
    end
  end

  defp add_sentence(sentence, {pos, chunks}, tokenizer) do
    case Tokenizer.count(tokenizer, sentence) do
      {:ok, token_count} ->
        end_byte = pos + byte_size(sentence)

        chunk = %Chunk{
          text: sentence,
          start_byte: pos,
          end_byte: end_byte,
          token_count: token_count
        }

        {:cont, {end_byte, [chunk | chunks]}}

      {:error, _reason} = error ->
        {:halt, error}
    end
  end

  defp create_chunks(sentences, tokenizer, config) do
    sentences = List.to_tuple(sentences)
    do_create_chunks(sentences, tuple_size(sentences), tokenizer, config, [], 0)
  end

  defp do_create_chunks(_sentences, total, _tokenizer, _config, sentence_chunks, pos)
       when pos >= total do
    {:ok, Enum.reverse(sentence_chunks)}
  end

  defp do_create_chunks(sentences, total, tokenizer, config, sentence_chunks, pos) do
    {chunk_sentences, split_idx} = split_at_chunk_boundary(sentences, total, pos, config)

    with {:ok, %SentenceChunk{} = sentence_chunk} <-
           create_sentence_chunk(chunk_sentences, tokenizer) do
      overlap_pos = find_overlap_start(chunk_sentences, split_idx, total, config)
      next_pos = max(overlap_pos, pos + 1)

      do_create_chunks(
        sentences,
        total,
        tokenizer,
        config,
        [sentence_chunk | sentence_chunks],
        next_pos
      )
    end
  end

  defp split_at_chunk_boundary(sentences, total, pos, config) do
    take_sentences(sentences, total, pos, config, [], 0, 0)
  end

  defp take_sentences(_sentences, total, pos, _config, acc, _tokens, _count)
       when pos >= total,
       do: {Enum.reverse(acc), pos}

  defp take_sentences(sentences, total, pos, config, acc, tokens, count) do
    sentence = elem(sentences, pos)
    new_tokens = tokens + sentence.token_count

    if tokens == 0 or
         new_tokens <= config.chunk_size or
         count < config.min_sentences_per_chunk do
      take_sentences(sentences, total, pos + 1, config, [sentence | acc], new_tokens, count + 1)
    else
      {Enum.reverse(acc), pos}
    end
  end

  defp create_sentence_chunk(sentences, tokenizer) do
    text = Enum.map_join(sentences, "", & &1.text)

    with {:ok, token_count} <- Tokenizer.count(tokenizer, text) do
      {:ok,
       %SentenceChunk{
         text: text,
         start_byte: hd(sentences).start_byte,
         end_byte: List.last(sentences).end_byte,
         token_count: token_count,
         sentences: sentences
       }}
    end
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
