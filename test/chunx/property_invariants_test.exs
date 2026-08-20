defmodule Chunx.PropertyInvariantsTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Chunx.Chunker.{Semantic, Sentence, SentenceSplitter, Token, Word}
  alias Chunx.Chunker.Semantic.Sentences
  alias Chunx.Helper
  alias Chunx.Tokenizer, as: TokenizerBoundary

  setup_all do
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("gpt2")
    %{tokenizer: tokenizer}
  end

  property "sentence splitting preserves text and delimiter order" do
    check all(
            segments <- list_of(word(), min_length: 1, max_length: 30),
            delimiter <- member_of([".", "!", "?", "\n", "。", "||"])
          ) do
      text = Enum.join(segments, delimiter)
      splits = SentenceSplitter.split(text, [delimiter])

      assert Enum.join(splits) == text
      assert length(splits) == length(segments)

      splits
      |> Enum.drop(-1)
      |> Enum.each(&assert String.ends_with?(&1, delimiter))
    end
  end

  property "sentence grouping is the clamped neighborhood of each sentence" do
    check all(
            sentences <- list_of(sentence(), max_length: 20),
            window <- integer(0..8)
          ) do
      groups = Sentences.build_sentence_groups(sentences, window)

      expected =
        sentences
        |> Enum.with_index()
        |> Enum.map(fn {_sentence, index} ->
          first = max(index - window, 0)
          Enum.slice(sentences, first, index + window - first + 1) |> Enum.join()
        end)

      assert groups == expected
      assert length(groups) == length(sentences)
    end
  end

  property "combining short sentences never loses or reorders text" do
    check all(
            splits <- list_of(string(:printable, max_length: 30), max_length: 30),
            minimum <- integer(0..20)
          ) do
      combined = Sentences.combine_short_sentences(splits, minimum)
      assert Enum.join(combined) == Enum.join(splits)
    end
  end

  property "sentence indices identify an exact partition even when text repeats" do
    check all(parts <- list_of(word(), min_length: 1, max_length: 30)) do
      text = Enum.join(parts)
      indexed = Sentences.find_sentence_indices(text, parts)

      {expected, _end_byte} =
        Enum.map_reduce(parts, 0, fn part, start_byte ->
          end_byte = start_byte + byte_size(part)
          {{part, start_byte, end_byte}, end_byte}
        end)

      assert indexed == expected
    end
  end

  property "word and sentence chunkers reconstruct generated text without overlap", %{
    tokenizer: tokenizer
  } do
    check all(
            sentences <- list_of(sentence(), min_length: 1, max_length: 30),
            chunk_size <- integer(1..30),
            max_runs: 60
          ) do
      text = Enum.join(sentences, " ")

      assert {:ok, word_chunks} =
               Word.chunk(text, tokenizer, chunk_size: chunk_size, chunk_overlap: 0)

      assert {:ok, sentence_chunks} =
               Sentence.chunk(text, tokenizer,
                 chunk_size: chunk_size,
                 chunk_overlap: 0,
                 short_sentence_threshold: 1
               )

      assert_partition(word_chunks, text)
      assert_partition(sentence_chunks, text)
    end
  end

  property "float overlap is equivalent to its normalized integer value", %{
    tokenizer: tokenizer
  } do
    check all(
            words <- list_of(word(), min_length: 1, max_length: 40),
            chunk_size <- integer(1..40),
            percentage <- integer(0..99),
            max_runs: 60
          ) do
      text = Enum.join(words, " ")
      overlap = percentage / 100
      normalized = floor(overlap * chunk_size)

      assert Token.chunk(text, tokenizer,
               chunk_size: chunk_size,
               chunk_overlap: overlap
             ) ==
               Token.chunk(text, tokenizer,
                 chunk_size: chunk_size,
                 chunk_overlap: normalized
               )

      assert Word.chunk(text, tokenizer,
               chunk_size: chunk_size,
               chunk_overlap: overlap
             ) ==
               Word.chunk(text, tokenizer,
                 chunk_size: chunk_size,
                 chunk_overlap: normalized
               )
    end
  end

  property "semantic chunking preserves generated text and exact chunk metadata", %{
    tokenizer: tokenizer
  } do
    embedding_fun = fn groups -> Enum.map(groups, fn _ -> Nx.tensor([1.0, 0.0]) end) end

    check all(
            sentences <- list_of(sentence(), min_length: 2, max_length: 20),
            chunk_size <- integer(1..30),
            max_runs: 50
          ) do
      text = Enum.join(sentences, " ")

      assert {:ok, chunks} =
               Semantic.chunk(text, tokenizer, embedding_fun,
                 chunk_size: chunk_size,
                 threshold: 0.5,
                 min_chars_per_sentence: 0
               )

      assert_partition(chunks, text)

      Enum.each(chunks, fn chunk ->
        assert Enum.map_join(chunk.sentences, & &1.text) == chunk.text
        assert {:ok, chunk.token_count} == TokenizerBoundary.count(tokenizer, chunk.text)
      end)
    end
  end

  property "median and standard deviation are invariant under translation" do
    check all(
            values <- list_of(integer(-1_000..1_000), min_length: 1, max_length: 50),
            translation <- integer(-1_000..1_000)
          ) do
      translated = Enum.map(values, &(&1 + translation))

      assert Helper.median(translated) == Helper.median(values) + translation

      assert_in_delta(
        Helper.standard_deviation(translated),
        Helper.standard_deviation(values),
        1.0e-10
      )
    end
  end

  defp word, do: member_of(~w(alpha beta gamma delta echo same café 世界))

  defp sentence do
    gen all(words <- list_of(word(), min_length: 1, max_length: 6)) do
      Enum.join(words, " ") <> "."
    end
  end

  defp assert_partition(chunks, text) do
    assert chunks != []
    assert Enum.map_join(chunks, & &1.text) == text
    assert hd(chunks).start_byte == 0
    assert List.last(chunks).end_byte == byte_size(text)

    chunks
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.each(fn [left, right] -> assert left.end_byte == right.start_byte end)

    Enum.each(chunks, fn chunk ->
      assert chunk.end_byte > chunk.start_byte
      assert chunk.text == binary_part(text, chunk.start_byte, chunk.end_byte - chunk.start_byte)
    end)
  end
end
