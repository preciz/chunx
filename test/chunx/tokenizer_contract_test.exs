defmodule Chunx.FailingTokenizer do
  @behaviour Chunx.Tokenizer

  @impl true
  def offsets(reason, _text), do: {:error, reason}
end

defmodule Chunx.MalformedTokenizer do
  @behaviour Chunx.Tokenizer

  @impl true
  def offsets(:response, _text), do: :unexpected
  def offsets(:offset, _text), do: {:ok, [{-1, 1}]}
end

defmodule Chunx.TokenizerContractTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Chunx.Chunker.{Recursive, Semantic, Sentence, Token, Word}
  alias Chunx.Chunker.Semantic.Sentences
  alias Chunx.Tokenizer, as: TokenizerBoundary

  setup_all do
    {:ok, gpt2} = Tokenizers.Tokenizer.from_pretrained("gpt2")

    {:ok, distilbert} =
      Tokenizers.Tokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    %{tokenizers: [gpt2, distilbert], distilbert: distilbert}
  end

  test "all chunkers report content tokens and exclude special-token offsets", %{
    distilbert: tokenizer
  } do
    text = "Alpha beta. Gamma delta."
    embedding_fun = fn groups -> Enum.map(groups, fn _ -> Nx.tensor([1.0, 0.0]) end) end

    results = [
      Token.chunk(text, tokenizer, chunk_size: 100, chunk_overlap: 0),
      Word.chunk(text, tokenizer, chunk_size: 100, chunk_overlap: 0),
      Sentence.chunk(text, tokenizer, chunk_size: 100, chunk_overlap: 0),
      Recursive.chunk(text, tokenizer, chunk_size: 100),
      Semantic.chunk(text, tokenizer, embedding_fun, chunk_size: 100, min_sentences: 10)
    ]

    for {:ok, [chunk]} <- results do
      assert chunk.token_count == content_token_count(chunk.text, tokenizer)
    end

    assert Enum.all?(results, &match?({:ok, [_]}, &1))
  end

  test "all tokenizer-backed entry points return the tokenizer error unchanged" do
    tokenizer = {Chunx.FailingTokenizer, :tokenizer_unavailable}
    embedding_fun = fn _groups -> flunk("embedding must not run after tokenization fails") end

    assert {:error, :tokenizer_unavailable} = Token.chunk("Text.", tokenizer)
    assert {:error, :tokenizer_unavailable} = Word.chunk("Text.", tokenizer)
    assert {:error, :tokenizer_unavailable} = Sentence.chunk("Text.", tokenizer)
    assert {:error, :tokenizer_unavailable} = Recursive.chunk("Text.", tokenizer)

    assert {:error, :tokenizer_unavailable} =
             Semantic.chunk("Text.", tokenizer, embedding_fun)

    assert {:error, :tokenizer_unavailable} =
             Sentences.prepare_sentences("Text.", tokenizer, embedding_fun)
  end

  test "all text entry points reject invalid UTF-8 before doing downstream work" do
    text = <<255>>
    tokenizer = {Chunx.FailingTokenizer, :tokenizer_must_not_run}
    embedding_fun = fn _groups -> flunk("embedding must not run for invalid text") end
    error = {:error, {:invalid_text, :invalid_utf8}}

    assert Token.chunk(text, tokenizer) == error
    assert Word.chunk(text, tokenizer) == error
    assert Sentence.chunk(text, tokenizer) == error
    assert Recursive.chunk(text, tokenizer) == error
    assert Semantic.chunk(text, tokenizer, embedding_fun) == error
    assert Sentences.prepare_sentences(text, tokenizer, embedding_fun) == error
    assert TokenizerBoundary.offsets(tokenizer, text) == error
    assert TokenizerBoundary.count(tokenizer, text) == error
  end

  test "rejects invalid UTF-8 delimiters" do
    invalid_delimiter = <<255>>

    assert_raise ArgumentError, "delimiters must contain non-empty strings", fn ->
      Sentence.chunk("Text.", {Chunx.FailingTokenizer, :unused}, delimiters: [invalid_delimiter])
    end

    assert_raise ArgumentError,
                 "levels must contain delimiter lists, :whitespace, or :tokens",
                 fn ->
                   Recursive.chunk("Text.", {Chunx.FailingTokenizer, :unused},
                     levels: [[invalid_delimiter]]
                   )
                 end

    assert_raise ArgumentError, "delimiters must contain non-empty strings", fn ->
      Sentences.prepare_sentences(
        "Text.",
        {Chunx.FailingTokenizer, :unused},
        fn _ -> [] end,
        delimiters: [invalid_delimiter]
      )
    end
  end

  test "invalid tokenizer responses and offsets become tagged errors" do
    assert {:error, {:invalid_tokenizer_response, :unexpected}} =
             Token.chunk("Text.", {Chunx.MalformedTokenizer, :response})

    assert {:error, {:invalid_tokenizer_offset, {-1, 1}}} =
             Token.chunk("Text.", {Chunx.MalformedTokenizer, :offset})
  end

  test "coalesces repeated and overlapping offsets into byte-safe units" do
    assert TokenizerBoundary.units([{0, 3}, {0, 3}, {1, 4}, {4, 5}]) ==
             [{0, 4, 3}, {4, 5, 1}]
  end

  property "packs ordinary token units like the sliding-window reference model" do
    check all(
            unit_count <- integer(1..100),
            chunk_size <- integer(1..30),
            overlap <- integer(0..(chunk_size - 1))
          ) do
      units = Enum.map(0..(unit_count - 1), &{&1, &1 + 1, 1})
      step = chunk_size - overlap

      expected =
        0
        |> Stream.iterate(&(&1 + step))
        |> Enum.take_while(&(&1 < unit_count))
        |> Enum.map(&Enum.slice(units, &1, chunk_size))

      assert TokenizerBoundary.pack(units, chunk_size, overlap) == expected
    end
  end

  property "token and recursive chunks remain byte-safe across tokenizers and Unicode", %{
    tokenizers: tokenizers
  } do
    fragment =
      member_of(["plain", " café", " e\u0301", " 世界", " ࠀ", " 👩‍💻", " العربية", ". "])

    check all(
            fragments <- list_of(fragment, min_length: 1, max_length: 16),
            chunk_size <- integer(1..12),
            max_runs: 40
          ) do
      text = Enum.join(fragments)

      for tokenizer <- tokenizers do
        assert {:ok, token_count} = TokenizerBoundary.count(tokenizer, text)
        assert {:ok, offsets} = TokenizerBoundary.offsets(tokenizer, text)
        assert token_count == length(offsets)

        assert {:ok, token_chunks} =
                 Token.chunk(text, tokenizer, chunk_size: chunk_size, chunk_overlap: 0)

        assert {:ok, recursive_chunks} =
                 Recursive.chunk(text, tokenizer,
                   chunk_size: chunk_size,
                   levels: [:tokens]
                 )

        assert_byte_safe_chunks(token_chunks, text, tokenizer, chunk_size)
        assert_byte_safe_chunks(recursive_chunks, text, tokenizer, chunk_size)
        assert Enum.map_join(recursive_chunks, & &1.text) == text
      end
    end
  end

  defp assert_byte_safe_chunks(chunks, text, tokenizer, _chunk_size) do
    Enum.each(chunks, fn chunk ->
      assert chunk.end_byte > chunk.start_byte
      assert chunk.text == binary_part(text, chunk.start_byte, chunk.end_byte - chunk.start_byte)
      assert chunk.token_count == content_token_count(chunk.text, tokenizer)
      assert grapheme_boundary?(text, chunk.start_byte)
      assert grapheme_boundary?(text, chunk.end_byte)
    end)
  end

  defp grapheme_boundary?(text, byte) do
    Enum.reduce_while(String.graphemes(text), 0, fn grapheme, offset ->
      cond do
        offset == byte -> {:halt, true}
        offset > byte -> {:halt, false}
        true -> {:cont, offset + byte_size(grapheme)}
      end
    end) == true or byte == byte_size(text)
  end

  defp content_token_count(text, tokenizer) do
    {:ok, count} = TokenizerBoundary.count(tokenizer, text)
    count
  end
end
