defmodule Chunx.Chunker.Recursive do
  @moduledoc """
  Implements recursive text chunking from coarse document boundaries to tokens.

  Text is split using each configured level in order. Segments that still exceed
  `:chunk_size` are passed to the next level, while adjacent segments are merged
  whenever they fit. The default hierarchy tries paragraphs, sentences,
  punctuation, whitespace, and finally token boundaries.
  """

  @behaviour Chunx.Chunker

  alias Chunx.Chunk
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

  @type level :: [String.t()] | :whitespace | :tokens
  @type chunk_opts :: [
          chunk_size: pos_integer(),
          levels: [level()]
        ]

  @default_opts [chunk_size: 512, levels: @default_levels]

  @doc """
  Recursively splits text into chunks no larger than `:chunk_size` tokens.

  ## Options

    * `:chunk_size` - Maximum number of content tokens per chunk (default: 512).
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
  @spec chunk(binary(), Tokenizers.Tokenizer.t(), chunk_opts()) ::
          {:ok, [Chunk.t()]} | {:error, term()}
  def chunk(text, tokenizer, opts \\ []) when is_binary(text) do
    config = @default_opts |> Keyword.merge(opts) |> validate_config!()

    if String.trim(text) == "" do
      {:ok, []}
    else
      {:ok, recursive_chunk(text, tokenizer, config, config.levels, 0)}
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
    token_count = count_tokens(text, tokenizer)

    cond do
      token_count == 0 ->
        []

      token_count <= config.chunk_size ->
        [create_chunk(text, start_byte, token_count)]

      true ->
        split_oversized(text, tokenizer, config, levels, start_byte)
    end
  end

  defp split_oversized(text, tokenizer, config, [], start_byte) do
    split_by_tokens(text, tokenizer, config.chunk_size, start_byte)
  end

  defp split_oversized(text, tokenizer, config, [:tokens | _rest], start_byte) do
    split_by_tokens(text, tokenizer, config.chunk_size, start_byte)
  end

  defp split_oversized(text, tokenizer, config, [level | rest], start_byte) do
    text
    |> split_at_level(level)
    |> merge_splits(tokenizer, config.chunk_size)
    |> chunks_from_splits(tokenizer, config, rest, start_byte)
  end

  defp split_at_level(text, :whitespace), do: SentenceSplitter.split(text, [" "])
  defp split_at_level(text, delimiters), do: SentenceSplitter.split(text, delimiters)

  defp merge_splits(splits, tokenizer, chunk_size) do
    {merged, current} =
      Enum.reduce(splits, {[], ""}, fn split, {merged, current} ->
        candidate = current <> split

        cond do
          current == "" ->
            {merged, split}

          count_tokens(current, tokenizer) == 0 ->
            {merged, candidate}

          count_tokens(candidate, tokenizer) <= chunk_size ->
            {merged, candidate}

          true ->
            {[current | merged], split}
        end
      end)

    [current | merged]
    |> Enum.reverse()
    |> Enum.reject(&(&1 == ""))
  end

  defp chunks_from_splits(splits, tokenizer, config, rest, start_byte) do
    {chunks, _end_byte} =
      Enum.map_reduce(splits, start_byte, fn split, offset ->
        token_count = count_tokens(split, tokenizer)

        chunks =
          if token_count > config.chunk_size do
            split_oversized(split, tokenizer, config, rest, offset)
          else
            [create_chunk(split, offset, token_count)]
          end

        {chunks, offset + byte_size(split)}
      end)

    List.flatten(chunks)
  end

  defp split_by_tokens(text, tokenizer, chunk_size, start_byte) do
    valid_offsets = token_offsets(text, tokenizer)

    valid_offsets
    |> Enum.chunk_every(chunk_size)
    |> token_groups_to_chunks(text, tokenizer, start_byte)
  end

  defp token_groups_to_chunks(groups, text, tokenizer, start_byte) do
    next_starts =
      groups
      |> Enum.drop(1)
      |> Enum.map(fn [{start_offset, _end_offset} | _rest] -> start_offset end)
      |> Kernel.++([byte_size(text)])

    groups
    |> Enum.zip(next_starts)
    |> Enum.map_reduce(0, fn {_group, end_offset}, offset ->
      chunk_text = binary_part(text, offset, end_offset - offset)
      chunk = create_chunk(chunk_text, start_byte + offset, count_tokens(chunk_text, tokenizer))

      {chunk, end_offset}
    end)
    |> elem(0)
  end

  defp count_tokens(text, tokenizer), do: length(token_offsets(text, tokenizer))

  defp token_offsets(text, tokenizer) do
    {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, text)

    encoding
    |> Tokenizers.Encoding.get_offsets()
    |> Enum.reject(fn {start_offset, end_offset} -> start_offset == end_offset end)
  end

  defp create_chunk(text, start_byte, token_count) do
    Chunk.new(text, start_byte, start_byte + byte_size(text), token_count)
  end
end
