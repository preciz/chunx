defmodule Chunx.Chunker.Recursive do
  @moduledoc """
  Splits text through an ordered sequence of structural boundaries.

  Text is split using each configured level in order. Segments that still exceed
  `:chunk_size` are passed to the next level, while adjacent segments are merged
  whenever they fit. The default hierarchy tries paragraphs, sentences,
  punctuation, whitespace, and finally token boundaries.
  """

  @behaviour Chunx.Chunker

  alias Chunx.{Chunk, Tokenizer}
  alias Chunx.Chunker.SentenceSplitter

  @default_levels [
    ["\n\n", "\r\n", "\n", "\r"],
    [". ", "! ", "? "],
    [
      "{",
      "}",
      "\"",
      "[",
      "]",
      "<",
      ">",
      "(",
      ")",
      ":",
      ";",
      ",",
      "—",
      "|",
      "~",
      "-",
      "...",
      "`",
      "'"
    ],
    :whitespace,
    :tokens
  ]

  @type level :: nonempty_list(String.t()) | :whitespace | :tokens
  @type chunk_opts :: [
          chunk_size: pos_integer(),
          levels: nonempty_list(level())
        ]

  @default_opts [chunk_size: 512, levels: @default_levels]

  @doc """
  Recursively splits text toward the `:chunk_size` target.

  ## Options

    * `:chunk_size` - Target maximum content-token count (default: 512). An
      indivisible grapheme may exceed the target.
    * `:levels` - Ordered splitting levels. Each level is a non-empty list of
      delimiters, `:whitespace`, or `:tokens`. Token splitting is always used as
      a final fallback when custom levels are exhausted.

  ## Examples

      iex> {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("gpt2")
      iex> text = "First paragraph." <> <<10, 10>> <> "Second paragraph."
      iex> {:ok, chunks} = Chunx.Chunker.Recursive.chunk(text, tokenizer, chunk_size: 4)
      iex> Enum.map(chunks, & &1.text)
      ["First paragraph." <> <<10, 10>>, "Second paragraph."]

  """
  @spec chunk(binary(), Tokenizer.t(), chunk_opts()) ::
          {:ok, [Chunk.t()]} | {:error, term()}
  def chunk(text, tokenizer, opts \\ []) when is_binary(text) do
    config = @default_opts |> Keyword.merge(opts) |> validate_config!()

    if String.trim(text) == "" do
      {:ok, []}
    else
      recursive_chunk(text, tokenizer, config, config.levels, 0)
    end
  end

  defp validate_config!(opts) do
    chunk_size = Keyword.fetch!(opts, :chunk_size)
    levels = Keyword.fetch!(opts, :levels)

    validate_chunk_size!(chunk_size)
    validate_levels!(levels)

    %{chunk_size: chunk_size, levels: levels}
  end

  defp validate_chunk_size!(chunk_size) when is_integer(chunk_size) and chunk_size > 0, do: :ok

  defp validate_chunk_size!(_chunk_size),
    do: raise(ArgumentError, "chunk_size must be positive")

  defp validate_levels!(levels) when is_list(levels) and levels != [] do
    if Enum.all?(levels, &valid_level?/1),
      do: :ok,
      else: raise(ArgumentError, "levels must contain delimiter lists, :whitespace, or :tokens")
  end

  defp validate_levels!(_levels),
    do: raise(ArgumentError, "levels must be a non-empty list")

  defp valid_level?(:whitespace), do: true
  defp valid_level?(:tokens), do: true

  defp valid_level?(delimiters) when is_list(delimiters) and delimiters != [],
    do: Enum.all?(delimiters, &(is_binary(&1) and &1 != ""))

  defp valid_level?(_level), do: false

  defp recursive_chunk(text, tokenizer, config, levels, start_byte) do
    with {:ok, token_count} <- Tokenizer.count(tokenizer, text) do
      cond do
        token_count == 0 ->
          {:ok, []}

        token_count <= config.chunk_size ->
          {:ok, [create_chunk(text, start_byte, token_count)]}

        true ->
          split_oversized(text, tokenizer, config, levels, start_byte)
      end
    end
  end

  defp split_oversized(text, tokenizer, config, [], start_byte) do
    split_by_tokens(text, tokenizer, config.chunk_size, start_byte)
  end

  defp split_oversized(text, tokenizer, config, [:tokens | _rest], start_byte) do
    split_by_tokens(text, tokenizer, config.chunk_size, start_byte)
  end

  defp split_oversized(text, tokenizer, config, [level | rest], start_byte) do
    splits = split_at_level(text, level)

    with {:ok, splits} <- merge_splits(splits, tokenizer, config.chunk_size) do
      chunks_from_splits(splits, tokenizer, config, rest, start_byte)
    end
  end

  defp split_at_level(text, :whitespace), do: SentenceSplitter.split(text, [" "])
  defp split_at_level(text, delimiters), do: SentenceSplitter.split(text, delimiters)

  defp merge_splits(splits, tokenizer, chunk_size) do
    result =
      Enum.reduce_while(splits, {[], "", nil}, fn split, state ->
        case merge_split(split, state, tokenizer, chunk_size) do
          {:ok, state} -> {:cont, state}
          {:error, _reason} = error -> {:halt, error}
        end
      end)

    case result do
      {merged, current, _token_count} -> {:ok, Enum.reverse([current | merged])}
      {:error, _reason} = error -> error
    end
  end

  defp merge_split(split, {merged, "", _current_count}, _tokenizer, _chunk_size),
    do: {:ok, {merged, split, nil}}

  defp merge_split(split, {merged, current, current_count}, tokenizer, chunk_size) do
    candidate = current <> split

    with {:ok, current_count} <- current_count(current_count, current, tokenizer) do
      merge_candidate(
        split,
        merged,
        current,
        current_count,
        candidate,
        tokenizer,
        chunk_size
      )
    end
  end

  defp merge_candidate(_split, merged, _current, 0, candidate, _tokenizer, _chunk_size),
    do: {:ok, {merged, candidate, nil}}

  defp merge_candidate(split, merged, current, _count, candidate, tokenizer, chunk_size) do
    with {:ok, candidate_count} <- Tokenizer.count(tokenizer, candidate) do
      if candidate_count <= chunk_size,
        do: {:ok, {merged, candidate, candidate_count}},
        else: {:ok, {[current | merged], split, nil}}
    end
  end

  defp current_count(nil, current, tokenizer), do: Tokenizer.count(tokenizer, current)
  defp current_count(current_count, _current, _tokenizer), do: {:ok, current_count}

  defp chunks_from_splits(splits, tokenizer, config, rest, start_byte) do
    result =
      Enum.reduce_while(splits, {[], start_byte}, fn split, state ->
        add_split_chunks(split, state, tokenizer, config, rest)
      end)

    case result do
      {:error, _reason} = error -> error
      {chunk_groups, _end_byte} -> {:ok, chunk_groups |> Enum.reverse() |> List.flatten()}
    end
  end

  defp add_split_chunks(split, {chunk_groups, offset}, tokenizer, config, rest) do
    case Tokenizer.count(tokenizer, split) do
      {:ok, token_count} ->
        add_counted_split(split, token_count, chunk_groups, offset, tokenizer, config, rest)

      {:error, _reason} = error ->
        {:halt, error}
    end
  end

  defp add_counted_split(split, token_count, chunk_groups, offset, tokenizer, config, rest) do
    result =
      if token_count > config.chunk_size,
        do: split_oversized(split, tokenizer, config, rest, offset),
        else: {:ok, [create_chunk(split, offset, token_count)]}

    case result do
      {:ok, chunks} -> {:cont, {[chunks | chunk_groups], offset + byte_size(split)}}
      {:error, _reason} = error -> {:halt, error}
    end
  end

  defp split_by_tokens(text, tokenizer, chunk_size, start_byte) do
    with {:ok, offsets} <- Tokenizer.offsets(tokenizer, text) do
      offsets
      |> Tokenizer.units()
      |> Tokenizer.pack(chunk_size, 0)
      |> token_groups_to_chunks(text, tokenizer, start_byte)
    end
  end

  defp token_groups_to_chunks(groups, text, tokenizer, start_byte) do
    build_token_chunks(groups, text, tokenizer, start_byte, 0, [])
  end

  defp build_token_chunks([_group], text, tokenizer, start_byte, offset, chunks) do
    with {:ok, chunks} <-
           add_token_chunk(text, tokenizer, start_byte, offset, byte_size(text), chunks) do
      {:ok, Enum.reverse(chunks)}
    end
  end

  defp build_token_chunks(
         [_group, [{end_offset, _, _} | _] = next_group | rest],
         text,
         tokenizer,
         start_byte,
         offset,
         chunks
       ) do
    with {:ok, chunks} <-
           add_token_chunk(text, tokenizer, start_byte, offset, end_offset, chunks) do
      build_token_chunks([next_group | rest], text, tokenizer, start_byte, end_offset, chunks)
    end
  end

  defp add_token_chunk(text, tokenizer, start_byte, offset, end_offset, chunks) do
    chunk_text = binary_part(text, offset, end_offset - offset)

    with {:ok, token_count} <- Tokenizer.count(tokenizer, chunk_text) do
      {:ok, [create_chunk(chunk_text, start_byte + offset, token_count) | chunks]}
    end
  end

  defp create_chunk(text, start_byte, token_count) do
    Chunk.new(text, start_byte, start_byte + byte_size(text), token_count)
  end
end
