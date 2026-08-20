defmodule Chunx.Chunker.Semantic.Sentences do
  @moduledoc false

  alias Chunx.{Chunk, Tokenizer}
  alias Chunx.Chunker.SentenceSplitter

  @min_chars_per_sentence 12
  @delimiters [".", "!", "?", "\n"]
  @similarity_window 1

  @spec prepare_sentences(
          text :: binary(),
          tokenizer :: Tokenizer.t(),
          embedding_fun :: ([binary()] -> [Nx.Tensor.t()]),
          opts :: keyword()
        ) :: [Chunk.t()] | {:error, term()}
  def prepare_sentences(text, tokenizer, embedding_fun, opts \\ [])
      when is_binary(text) and is_function(embedding_fun, 1) do
    min_chars_per_sentence = Keyword.get(opts, :min_chars_per_sentence, @min_chars_per_sentence)
    delimiters = Keyword.get(opts, :delimiters, @delimiters)
    similarity_window = Keyword.get(opts, :similarity_window, @similarity_window)

    validate_options!(delimiters, min_chars_per_sentence, similarity_window)

    sentences = split_sentences(text, delimiters, min_chars_per_sentence)
    sentences_with_indices = find_sentence_indices(text, sentences)

    with {:ok, token_counts} <- get_token_counts(sentences, tokenizer) do
      sentence_groups = build_sentence_groups(sentences, similarity_window)
      embeddings = embedding_fun.(sentence_groups)

      if not is_list(embeddings) or length(embeddings) != length(sentence_groups) do
        raise ArgumentError,
              "embedding_fun must return one embedding for each sentence group"
      end

      sentences_with_indices
      |> Enum.zip(token_counts)
      |> Enum.zip(embeddings)
      |> Enum.map(fn {{{text, start_byte, end_byte}, token_count}, embedding} ->
        %Chunk{
          text: text,
          start_byte: start_byte,
          end_byte: end_byte,
          token_count: token_count,
          embedding: embedding
        }
      end)
    end
  end

  @spec find_sentence_indices(binary(), [binary()]) ::
          [{binary(), non_neg_integer(), non_neg_integer()}]
  def find_sentence_indices(text, sentences) do
    {sentences_with_indices, _} =
      Enum.reduce(sentences, {[], 0}, fn sentence, {acc, current_idx} ->
        case :binary.match(text, sentence, scope: {current_idx, byte_size(text) - current_idx}) do
          {pos, _len} ->
            start_idx = pos
            end_idx = pos + byte_size(sentence)
            {[{sentence, start_idx, end_idx} | acc], end_idx}

          :nomatch ->
            start_idx = current_idx
            end_idx = current_idx + byte_size(sentence)
            {[{sentence, start_idx, end_idx} | acc], end_idx}
        end
      end)

    Enum.reverse(sentences_with_indices)
  end

  @spec split_sentences(binary(), [binary()], non_neg_integer()) :: [binary()]
  def split_sentences(text, delimiters, min_chars_per_sentence) do
    text
    |> SentenceSplitter.split(delimiters)
    |> combine_short_sentences(min_chars_per_sentence)
  end

  @spec combine_short_sentences([binary()], non_neg_integer()) :: [binary()]
  def combine_short_sentences(splits, min_chars) do
    {sentences, current} =
      Enum.reduce(splits, {[], ""}, fn split, {sentences, current} ->
        if String.length(String.trim(split)) < min_chars do
          {sentences, current <> split}
        else
          append_valid_sentence(sentences, current, split)
        end
      end)

    sentences = if current != "", do: [current | sentences], else: sentences
    Enum.reverse(sentences)
  end

  defp append_valid_sentence(sentences, "", split) do
    {sentences, split}
  end

  defp append_valid_sentence(sentences, current, split) do
    {[current | sentences], split}
  end

  defp get_token_counts(sentences, tokenizer) do
    result = Enum.reduce_while(sentences, [], &add_token_count(&1, &2, tokenizer))

    case result do
      token_counts when is_list(token_counts) -> {:ok, Enum.reverse(token_counts)}
      {:error, _reason} = error -> error
    end
  end

  defp add_token_count(sentence, token_counts, tokenizer) do
    case Tokenizer.count(tokenizer, sentence) do
      {:ok, token_count} -> {:cont, [token_count | token_counts]}
      {:error, _reason} = error -> {:halt, error}
    end
  end

  @spec build_sentence_groups([binary()], non_neg_integer()) :: [binary()]
  def build_sentence_groups(sentences, 0), do: sentences

  def build_sentence_groups([], similarity_window)
      when is_integer(similarity_window) and similarity_window > 0,
      do: []

  def build_sentence_groups(sentences, similarity_window)
      when is_integer(similarity_window) and similarity_window > 0 do
    sentence_tuple = List.to_tuple(sentences)
    last_index = tuple_size(sentence_tuple) - 1

    Enum.map(0..last_index, fn index ->
      first = max(0, index - similarity_window)
      last = min(last_index, index + similarity_window)
      Enum.map_join(first..last, &elem(sentence_tuple, &1))
    end)
  end

  defp validate_options!(delimiters, min_chars_per_sentence, similarity_window) do
    validate_delimiters!(delimiters)
    validate_non_negative_integer!(min_chars_per_sentence, "min_chars_per_sentence")
    validate_non_negative_integer!(similarity_window, "similarity_window")
  end

  defp validate_delimiters!(delimiters) when is_list(delimiters) and delimiters != [] do
    if Enum.all?(delimiters, &(is_binary(&1) and &1 != "")),
      do: :ok,
      else: raise(ArgumentError, "delimiters must contain non-empty strings")
  end

  defp validate_delimiters!(_delimiters),
    do: raise(ArgumentError, "delimiters must contain non-empty strings")

  defp validate_non_negative_integer!(value, _name)
       when is_integer(value) and value >= 0,
       do: :ok

  defp validate_non_negative_integer!(_value, name),
    do: raise(ArgumentError, "#{name} must be a non-negative integer")
end
