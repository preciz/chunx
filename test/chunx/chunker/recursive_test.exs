defmodule Chunx.Chunker.RecursiveTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  doctest Chunx.Chunker.Recursive

  alias Chunx.Chunk
  alias Chunx.Chunker.Recursive

  @sample_text """
  # Recursive chunking

  Recursive chunking starts with broad document structure. It keeps paragraphs together when they fit.

  Oversized paragraphs are divided into sentences. Long sentences then fall back to punctuation, words, and tokens.

  This hierarchy preserves useful context while respecting the selected token limit.
  """

  setup_all do
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("gpt2")
    %{tokenizer: tokenizer}
  end

  describe "chunk/3" do
    test "handles empty and whitespace-only text", %{tokenizer: tokenizer} do
      assert {:ok, []} = Recursive.chunk("", tokenizer)
      assert {:ok, []} = Recursive.chunk(" \n\t ", tokenizer)
    end

    test "handles non-text content ignored by the tokenizer" do
      {:ok, tokenizer} =
        Tokenizers.Tokenizer.from_pretrained("distilbert/distilbert-base-uncased")

      assert {:ok, []} = Recursive.chunk(<<0>>, tokenizer)
    end

    test "keeps ignored content beside tokenized text" do
      {:ok, tokenizer} =
        Tokenizers.Tokenizer.from_pretrained("distilbert/distilbert-base-uncased")

      text = <<0>> <> " alpha beta"

      assert {:ok, chunks} =
               Recursive.chunk(text, tokenizer,
                 chunk_size: 1,
                 levels: [:whitespace, :tokens]
               )

      assert_valid_chunks(chunks, text, tokenizer, 1)
    end

    test "returns a short input as one chunk", %{tokenizer: tokenizer} do
      assert {:ok, [%Chunk{} = chunk]} = Recursive.chunk("A", tokenizer, chunk_size: 1)
      assert chunk.text == "A"
      assert chunk.start_byte == 0
      assert chunk.end_byte == 1
      assert chunk.token_count == 1
    end

    test "recursively chunks structured text", %{tokenizer: tokenizer} do
      assert {:ok, chunks} = Recursive.chunk(@sample_text, tokenizer, chunk_size: 12)

      assert length(chunks) > 1
      assert_valid_chunks(chunks, @sample_text, tokenizer, 12)
    end

    test "keeps paragraph boundaries when they fit", %{tokenizer: tokenizer} do
      text = "First paragraph.\n\nSecond paragraph."

      assert {:ok, chunks} = Recursive.chunk(text, tokenizer, chunk_size: 4)
      assert Enum.map(chunks, & &1.text) == ["First paragraph.\n\n", "Second paragraph."]
      assert_valid_chunks(chunks, text, tokenizer, 4)
    end

    test "falls through to whitespace boundaries", %{tokenizer: tokenizer} do
      text = "one two three four five"

      assert {:ok, chunks} =
               Recursive.chunk(text, tokenizer,
                 chunk_size: 3,
                 levels: [:whitespace, :tokens]
               )

      assert Enum.map(chunks, & &1.text) == ["one two ", "three four five"]
      assert_valid_chunks(chunks, text, tokenizer, 3)
    end

    test "uses token boundaries for an unbroken segment", %{tokenizer: tokenizer} do
      text = String.duplicate("antidisestablishmentarianism", 8)

      assert {:ok, chunks} =
               Recursive.chunk(text, tokenizer,
                 chunk_size: 5,
                 levels: [["\n\n"]]
               )

      assert length(chunks) > 1
      assert_valid_chunks(chunks, text, tokenizer, 5)
    end

    test "supports custom Unicode sentence delimiters", %{tokenizer: tokenizer} do
      text = "自然言語処理です。検索に使います！文脈を保ちます？最後の文です。"

      assert {:ok, chunks} =
               Recursive.chunk(text, tokenizer,
                 chunk_size: 12,
                 levels: [["。", "！", "？"], :tokens]
               )

      assert length(chunks) > 1
      assert_valid_chunks(chunks, text, tokenizer, 12)
    end

    test "preserves repeated text offsets", %{tokenizer: tokenizer} do
      text = "same sentence. same sentence. same sentence. same sentence."

      assert {:ok, chunks} = Recursive.chunk(text, tokenizer, chunk_size: 4)

      assert_valid_chunks(chunks, text, tokenizer, 4)

      assert chunks
             |> Enum.map(& &1.start_byte)
             |> Enum.chunk_every(2, 1, :discard)
             |> Enum.all?(fn [left, right] -> left < right end)
    end

    test "validates configuration", %{tokenizer: tokenizer} do
      assert_raise ArgumentError, "chunk_size must be positive", fn ->
        Recursive.chunk("text", tokenizer, chunk_size: 0)
      end

      assert_raise ArgumentError, "levels must be a non-empty list", fn ->
        Recursive.chunk("text", tokenizer, levels: [])
      end

      assert_raise ArgumentError,
                   "levels must contain delimiter lists, :whitespace, or :tokens",
                   fn ->
                     Recursive.chunk("text", tokenizer, levels: [:unknown])
                   end

      assert_raise ArgumentError,
                   "levels must contain delimiter lists, :whitespace, or :tokens",
                   fn ->
                     Recursive.chunk("text", tokenizer, levels: [[""]])
                   end
    end
  end

  property "reconstructs generated text with exact byte offsets", %{tokenizer: tokenizer} do
    word = member_of(~w(alpha beta gamma delta epsilon zeta))

    check all(
            words <- list_of(word, min_length: 1, max_length: 80),
            chunk_size <- integer(1..20)
          ) do
      text = Enum.join(words, " ")
      assert {:ok, chunks} = Recursive.chunk(text, tokenizer, chunk_size: chunk_size)
      assert_valid_chunks(chunks, text, tokenizer, chunk_size)
    end
  end

  defp assert_valid_chunks(chunks, text, tokenizer, chunk_size) do
    assert chunks != []
    assert Enum.map_join(chunks, & &1.text) == text

    Enum.each(chunks, fn chunk ->
      assert chunk.text == binary_part(text, chunk.start_byte, chunk.end_byte - chunk.start_byte)
      assert chunk.start_byte >= 0
      assert chunk.end_byte > chunk.start_byte
      assert chunk.token_count == content_token_count(chunk.text, tokenizer)
      assert chunk.token_count <= chunk_size
    end)
  end

  defp content_token_count(text, tokenizer) do
    {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, text)

    encoding
    |> Tokenizers.Encoding.get_offsets()
    |> Enum.count(fn {start_offset, end_offset} -> start_offset != end_offset end)
  end
end
