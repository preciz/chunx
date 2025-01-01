defmodule Chunx.Chunker.Semantic do
  @moduledoc """
  Implements semantic chunking using sentence embeddings.

  Splits text into semantically coherent chunks using embeddings
  while respecting token limits.
  """

  @behaviour Chunx.Chunker

  alias Chunx.SentenceChunk
  alias Scholar.Metrics.Distance

  @type chunk_opts :: [
          chunk_size: pos_integer(),
          threshold: float() | :auto,
          min_sentences: pos_integer(),
          min_chunk_size: pos_integer(),
          threshold_step: float()
        ]

  @default_opts [
    chunk_size: 512,
    threshold: :auto,
    min_sentences: 1,
    min_chunk_size: 2,
    threshold_step: 0.01
  ]

  @doc """
  Splits text into semantically coherent chunks using embeddings.

  ## Options
    * `:chunk_size` - Maximum number of tokens per chunk (default: 512)
    * `:threshold` - Threshold for semantic similarity (0-1) or :auto (default: :auto)
    * `:min_sentences` - Minimum number of sentences per chunk (default: 1)
    * `:min_chunk_size` - Minimum number of tokens per chunk (default: 2)
    * `:threshold_step` - Step size for threshold calculation (default: 0.01)
  """
  @spec chunk(
          binary(),
          Tokenizers.Tokenizer.t(),
          (list(String.t()) -> list(Nx.Tensor.t())),
          chunk_opts()
        ) ::
          {:ok, [SentenceChunk.t()]} | {:error, term()}
  def chunk(text, tokenizer, embedding_fun, opts \\ [])
      when is_binary(text) and is_function(embedding_fun, 1) do
    opts = Keyword.merge(@default_opts, opts)
    config = validate_config!(opts)

    if String.trim(text) == "" do
      {:ok, []}
    else
      sentences =
        Chunx.Chunker.Semantic.Sentences.prepare_sentences(text, tokenizer, embedding_fun, opts)

      if length(sentences) <= config.min_sentences do
        chunk = create_chunk(sentences)
        {:ok, [chunk]}
      else
        {threshold, maybe_pairwise_similarities} =
          calculate_similarity_threshold(sentences, config)

        sentence_groups =
          group_sentences(sentences, threshold, maybe_pairwise_similarities, config)

        chunks = split_chunks(sentence_groups, config)
        {:ok, chunks}
      end
    end
  end

  defp validate_config!(opts) do
    chunk_size = Keyword.fetch!(opts, :chunk_size)
    threshold = Keyword.fetch!(opts, :threshold)
    min_sentences = Keyword.fetch!(opts, :min_sentences)
    min_chunk_size = Keyword.fetch!(opts, :min_chunk_size)
    threshold_step = Keyword.fetch!(opts, :threshold_step)

    if chunk_size <= 0, do: raise(ArgumentError, "chunk_size must be positive")
    if min_sentences <= 0, do: raise(ArgumentError, "min_sentences must be positive")
    if min_chunk_size <= 0, do: raise(ArgumentError, "min_chunk_size must be positive")

    if threshold_step <= 0 or threshold_step >= 1,
      do: raise(ArgumentError, "threshold_step must be between 0 and 1")

    case threshold do
      :auto -> :ok
      t when is_float(t) and t >= 0 and t <= 1 -> :ok
      _ -> raise(ArgumentError, "threshold must be :auto or a float between 0 and 1")
    end

    %{
      chunk_size: chunk_size,
      threshold: threshold,
      min_sentences: min_sentences,
      min_chunk_size: min_chunk_size,
      threshold_step: threshold_step
    }
  end

  defp calculate_similarity_threshold(sentences, %{threshold: :auto} = config) do
    calculate_threshold_via_binary_search(sentences, config)
  end

  defp calculate_similarity_threshold(_sentences, %{threshold: threshold}) do
    {threshold, nil}
  end

  defp calculate_threshold_via_binary_search(sentences, config) do
    similarities = compute_pairwise_similarities(sentences)
    similarity_values = Enum.map(similarities, fn {_, _, sim} -> sim end)
    median = Chunx.Helper.median(similarity_values)
    std = Chunx.Helper.standard_deviation(similarity_values)

    low = max(median - std, 0.0)
    high = min(median + std, 1.0)

    {find_optimal_threshold(sentences, config, low, high), similarities}
  end

  defp find_optimal_threshold(sentences, config, low, high, iterations \\ 0)

  defp find_optimal_threshold(_, config, low, high, _iterations)
       when abs(high - low) <= config.threshold_step do
    (low + high) / 2
  end

  defp find_optimal_threshold(_, _, low, high, iterations) when iterations > 10 do
    (low + high) / 2
  end

  defp find_optimal_threshold(sentences, config, low, high, iterations) do
    threshold = (low + high) / 2
    split_ranges = get_split_indices(sentences, threshold, config)

    token_counts =
      split_ranges
      |> Enum.map(fn [start_idx, end_idx] ->
        sentences
        |> Enum.slice(start_idx, end_idx - start_idx)
        |> Enum.map(& &1.token_count)
        |> Enum.sum()
      end)

    all_valid_size =
      Enum.all?(token_counts, &(&1 >= config.min_chunk_size and &1 <= config.chunk_size))

    cond do
      all_valid_size ->
        threshold

      Enum.any?(token_counts, &(&1 > config.chunk_size)) ->
        # If any chunk is too large, increase threshold to create smaller chunks
        find_optimal_threshold(
          sentences,
          config,
          threshold + config.threshold_step,
          high,
          iterations + 1
        )

      true ->
        # If chunks are too small, decrease threshold to create larger chunks
        find_optimal_threshold(
          sentences,
          config,
          low,
          threshold - config.threshold_step,
          iterations + 1
        )
    end
  end

  defp compute_pairwise_similarities(sentences) do
    sentences = Enum.with_index(sentences)

    sentences
    |> Enum.zip(Enum.drop(sentences, 1))
    |> Enum.map(fn {{s1, i}, {s2, j}} ->
      {i, j, cosine_similarity(s1.embedding, s2.embedding)}
    end)
  end

  defp cosine_similarity(v1, v2) do
    1 - Nx.to_number(Distance.cosine(v1, v2))
  end

  defp group_sentences(sentences, threshold, maybe_pairwise_similarities, config) do
    similarities = maybe_pairwise_similarities || compute_pairwise_similarities(sentences)

    # Average similarities for each sentence with its neighbors
    all_similarities =
      similarities
      |> Enum.flat_map(fn {i, j, similarity} ->
        [{i, similarity}, {j, similarity}]
      end)
      |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))

    avg_similarities =
      similarities
      |> Enum.map(fn {i, _, _} ->
        list = all_similarities[i]

        Enum.sum(list) / length(list)
      end)

    # Get split points based on similarity threshold
    split_indices = get_split_indices(avg_similarities, threshold, config)

    # Create groups based on split indices
    split_indices
    |> Enum.map(fn [start_idx, end_idx] ->
      Enum.slice(sentences, start_idx..end_idx)
    end)
  end

  defp get_split_indices(avg_similarities, threshold, config) do
    # Get indices where similarity drops below threshold
    split_points =
      avg_similarities
      |> Enum.with_index()
      |> Enum.filter(fn {sim, _idx} -> sim <= threshold end)
      |> Enum.map(fn {_sim, idx} -> idx + 1 end)

    # Add start and end points
    splits = [0] ++ split_points ++ [length(avg_similarities) + 1]

    # Filter splits that don't meet minimum sentence requirement
    splits
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.filter(fn [start_idx, end_idx] ->
      end_idx - start_idx >= config.min_sentences
    end)
  end

  defp split_chunks(sentence_groups, config) do
    sentence_groups
    |> Enum.flat_map(fn group ->
      split_group_into_chunks(group, config)
    end)
  end

  defp split_group_into_chunks(group, config) do
    {chunks, current_chunk, _current_tokens} =
      group
      |> Enum.reduce({[], [], 0}, fn sentence, {chunks, current_chunk, current_tokens} ->
        new_tokens = current_tokens + sentence.token_count

        cond do
          new_tokens <= config.chunk_size ->
            {chunks, current_chunk ++ [sentence], new_tokens}

          length(current_chunk) < config.min_sentences ->
            # If the current chunk doesn't meet min_sentences, force add sentence
            {chunks, current_chunk ++ [sentence], new_tokens}

          true ->
            chunk = create_chunk(current_chunk)
            {chunks ++ [chunk], [sentence], sentence.token_count}
        end
      end)

    if current_chunk != [], do: chunks ++ [create_chunk(current_chunk)], else: chunks
  end

  defp create_chunk(sentences) do
    text = Enum.map_join(sentences, "", & &1.text)
    token_count = Enum.sum(Enum.map(sentences, & &1.token_count))
    start_byte = hd(sentences).start_byte
    end_byte = List.last(sentences).end_byte

    %SentenceChunk{
      text: text,
      start_byte: start_byte,
      end_byte: end_byte,
      token_count: token_count,
      sentences: sentences
    }
  end
end
