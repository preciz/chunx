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
        {threshold, avg_similarities} =
          calculate_similarity_threshold(sentences, config)

        sentence_groups =
          group_sentences(sentences, threshold, avg_similarities, config)

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

  defp calculate_similarity_threshold(sentences, %{threshold: threshold}) do
    {threshold, compute_avg_similarities(sentences)}
  end

  defp compute_avg_similarities(sentences) do
    similarities = compute_pairwise_similarities(sentences)

    all_similarities =
      similarities
      |> Enum.flat_map(fn {i, j, similarity} ->
        [{i, similarity}, {j, similarity}]
      end)
      |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))

    0..(length(sentences) - 1)
    |> Enum.map(fn i ->
      list = all_similarities[i]
      Enum.sum(list) / length(list)
    end)
  end

  defp calculate_threshold_via_binary_search(sentences, config) do
    similarities = compute_pairwise_similarities(sentences)

    all_similarities =
      similarities
      |> Enum.flat_map(fn {i, j, similarity} ->
        [{i, similarity}, {j, similarity}]
      end)
      |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))

    avg_similarities =
      0..(length(sentences) - 1)
      |> Enum.map(fn i ->
        list = all_similarities[i]
        Enum.sum(list) / length(list)
      end)

    similarity_values = Enum.map(similarities, fn {_, _, sim} -> sim end)
    median = Chunx.Helper.median(similarity_values)
    std = Chunx.Helper.standard_deviation(similarity_values)

    low = max(median - std, 0.0)
    high = min(median + std, 1.0)

    {cumulative_tokens_list, _} =
      Enum.map_reduce(sentences, 0, fn s, acc ->
        new_acc = acc + s.token_count
        {new_acc, new_acc}
      end)

    cumulative_tokens = List.to_tuple([0 | cumulative_tokens_list])

    avg_sims_tuple = List.to_tuple(avg_similarities)

    optimal_threshold =
      find_optimal_threshold(
        avg_sims_tuple,
        cumulative_tokens,
        length(sentences),
        config,
        low,
        high
      )

    {optimal_threshold, avg_similarities}
  end

  defp find_optimal_threshold(
         avg_similarities,
         cumulative_tokens,
         total_sentences,
         config,
         low,
         high,
         iterations \\ 0
       ) do
    if abs(high - low) <= config.threshold_step or iterations > 10 do
      (low + high) / 2
    else
      threshold = (low + high) / 2
      split_ranges = get_split_indices(avg_similarities, total_sentences, threshold, config)

      token_counts =
        Enum.map(split_ranges, fn [start_idx, end_idx] ->
          end_tokens = elem(cumulative_tokens, end_idx)
          start_tokens = elem(cumulative_tokens, start_idx)
          end_tokens - start_tokens
        end)

      all_valid_size =
        Enum.all?(token_counts, &(&1 >= config.min_chunk_size and &1 <= config.chunk_size))

      cond do
        all_valid_size ->
          threshold

        Enum.any?(token_counts, &(&1 > config.chunk_size)) ->
          find_optimal_threshold(
            avg_similarities,
            cumulative_tokens,
            total_sentences,
            config,
            threshold + config.threshold_step,
            high,
            iterations + 1
          )

        true ->
          find_optimal_threshold(
            avg_similarities,
            cumulative_tokens,
            total_sentences,
            config,
            low,
            threshold - config.threshold_step,
            iterations + 1
          )
      end
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

  defp group_sentences(sentences, threshold, avg_similarities, config) do
    split_indices =
      get_split_indices(List.to_tuple(avg_similarities), length(sentences), threshold, config)

    split_indices
    |> Enum.map(fn [start_idx, end_idx] ->
      Enum.slice(sentences, start_idx..(end_idx - 1))
    end)
    |> Enum.reject(&Enum.empty?/1)
  end

  defp get_split_indices(avg_similarities, total_sentences, threshold, config) do
    # avg_similarities is a Tuple for fast iteration
    split_points =
      0..(total_sentences - 1)
      |> Enum.filter(fn idx -> elem(avg_similarities, idx) <= threshold end)
      |> Enum.map(&(&1 + 1))

    splits = [0] ++ split_points ++ [total_sentences]

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
    {chunks, current_chunk, _current_tokens, _current_length} =
      group
      |> Enum.reduce({[], [], 0, 0}, fn sentence,
                                        {chunks, current_chunk, current_tokens, current_length} ->
        new_tokens = current_tokens + sentence.token_count

        if new_tokens <= config.chunk_size or current_length < config.min_sentences do
          {chunks, [sentence | current_chunk], new_tokens, current_length + 1}
        else
          chunk = create_chunk(Enum.reverse(current_chunk))
          {[chunk | chunks], [sentence], sentence.token_count, 1}
        end
      end)

    [create_chunk(Enum.reverse(current_chunk)) | chunks]
    |> Enum.reverse()
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
