defmodule Chunx.Chunker.Semantic.Sentences do
  @moduledoc """
  Handles sentence preparation and processing for semantic chunking.
  """

  alias Chunx.Chunk

  @separator "🦛"
  @min_chars_per_sentence 12
  @delimiters [".", "!", "?", "\n"]
  @similarity_window 1

  @doc """
  Prepares sentences from text with tokenization and embeddings.
  """
  @spec prepare_sentences(
          text :: binary(),
          tokenizer :: Tokenizers.Tokenizer.t(),
          embedding_fun :: (list(binary()) -> list(Nx.Tensor.t())),
          opts :: keyword()
        ) :: list(Chunk.t())
  def prepare_sentences(text, tokenizer, embedding_fun, opts \\ [])
      when is_binary(text) and is_function(embedding_fun, 1) do
    separator = Keyword.get(opts, :separator, @separator)
    min_chars_per_sentence = Keyword.get(opts, :min_chars_per_sentence, @min_chars_per_sentence)
    delimiters = Keyword.get(opts, :delimiters, @delimiters)
    similarity_window = Keyword.get(opts, :similarity_window, @similarity_window)

    sentences = split_sentences(text, separator, delimiters, min_chars_per_sentence)
    sentences_with_indices = find_sentence_indices(text, sentences)
    token_counts = get_token_counts(sentences, tokenizer)
    sentence_groups = build_sentence_groups(sentences, similarity_window)
    embeddings = embedding_fun.(sentence_groups)

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

  @spec find_sentence_indices(binary(), list(binary())) ::
          list({binary(), non_neg_integer(), non_neg_integer()})
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

  @spec split_sentences(binary(), binary(), list(binary()), non_neg_integer()) :: list(binary())
  def split_sentences(text, separator, delimiters, min_chars_per_sentence) do
    text_with_sep =
      Enum.reduce(delimiters, text, fn delimiter, acc ->
        String.replace(acc, delimiter, delimiter <> separator)
      end)

    initial_splits =
      text_with_sep
      |> String.split(separator)
      |> Enum.reject(&(&1 == ""))

    combine_short_sentences(initial_splits, min_chars_per_sentence)
  end

  @spec combine_short_sentences(list(binary()), non_neg_integer()) :: list(binary())
  def combine_short_sentences(splits, min_chars) do
    {sentences, current} =
      Enum.reduce(splits, {[], ""}, fn split, {sentences, current} ->
        if String.length(String.trim(split)) < min_chars do
          {sentences, current <> split}
        else
          if current != "" do
            {sentences ++ [current], split}
          else
            {sentences, split}
          end
        end
      end)

    if current != "" do
      sentences ++ [current]
    else
      sentences
    end
  end

  defp get_token_counts(sentences, tokenizer) do
    sentences
    |> Enum.map(fn sentence ->
      {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, sentence)
      Tokenizers.Encoding.get_length(encoding)
    end)
  end

  @spec build_sentence_groups(list(binary()), non_neg_integer()) :: list(binary())
  def build_sentence_groups(sentences, 0), do: sentences

  def build_sentence_groups(sentences, similarity_window) when is_integer(similarity_window) do
    len = length(sentences)

    sentences
    |> Enum.with_index()
    |> Enum.map(fn {_sentence, index} ->
      sentences
      |> Enum.slice(max(0, index - similarity_window)..min(len - 1, index + similarity_window))
      |> Enum.join()
    end)
  end
end
