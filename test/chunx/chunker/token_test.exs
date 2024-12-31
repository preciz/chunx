defmodule Chunx.Chunker.TokenTest do
  use ExUnit.Case, async: true
  doctest Chunx.Chunker.Token
  alias Chunx.Chunker.Token
  alias Chunx.Chunk

  setup do
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    {:ok, tokenizer: tokenizer}
  end

  @sample_text """
  The process of text chunking in RAG applications represents a delicate balance between competing requirements.
  On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context
  that can be understood and processed independently. On the other, we must optimize for information density,
  ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy.
  """

  @complex_markdown """
  # Heading 1
  This is a paragraph with some **bold text** and _italic text_.
  ## Heading 2
  - Bullet point 1
  - Bullet point 2 with `inline code`
  ```elixir
  # Code block
  def hello_world do
    IO.puts("Hello, world!")
  end
  ```
  Another paragraph with [a link](https://example.com) and an image:
  ![Alt text](https://example.com/image.jpg)
  > A blockquote with multiple lines
  > that spans more than one line.
  Finally, a paragraph at the end.
  """

  describe "chunk/3" do
    test "handles empty text", %{tokenizer: tokenizer} do
      assert {:ok, []} = Token.chunk("", tokenizer)
    end

    test "creates chunks with default options", %{tokenizer: tokenizer} do
      text = String.duplicate("Hello world. ", 1000)
      {:ok, chunks} = Token.chunk(text, tokenizer)

      assert length(chunks) > 1

      assert Enum.all?(chunks, fn chunk ->
               is_binary(chunk.text) and
                 chunk.token_count > 0 and
                 chunk.start_index >= 0 and
                 chunk.end_index > chunk.start_index
             end)
    end

    test "respects chunk size", %{tokenizer: tokenizer} do
      text = String.duplicate("Hello world. ", 100)
      chunk_size = 10
      {:ok, chunks} = Token.chunk(text, tokenizer, chunk_size: chunk_size, chunk_overlap: 2)

      assert Enum.all?(chunks, fn chunk ->
               chunk.token_count <= chunk_size
             end)
    end

    test "handles overlap correctly", %{tokenizer: tokenizer} do
      text = String.duplicate("Hello world. ", 100)
      {:ok, chunks} = Token.chunk(text, tokenizer, chunk_size: 10, chunk_overlap: 5)

      # Check that consecutive chunks have overlapping content
      chunks
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.each(fn [chunk1, chunk2] ->
        assert chunk2.start_index < chunk1.end_index
      end)
    end

    test "validates chunk_size", %{tokenizer: tokenizer} do
      assert_raise ArgumentError, "chunk_size must be positive", fn ->
        Token.chunk("test", tokenizer, chunk_size: 0)
      end
    end

    test "validates overlap as integer", %{tokenizer: tokenizer} do
      assert_raise ArgumentError, "chunk_overlap must be less than chunk_size", fn ->
        Token.chunk("test", tokenizer, chunk_size: 10, chunk_overlap: 10)
      end
    end

    test "validates overlap as float", %{tokenizer: tokenizer} do
      assert_raise ArgumentError, "chunk_overlap percentage must be less than 1", fn ->
        Token.chunk("test", tokenizer, chunk_size: 10, chunk_overlap: 1.0)
      end
    end

    test "handles special tokens correctly", %{tokenizer: tokenizer} do
      # Text with characters that might trigger special tokens
      text = "[CLS] Hello [SEP] World [PAD]"
      {:ok, chunks} = Token.chunk(text, tokenizer, chunk_size: 5, chunk_overlap: 1)

      assert Enum.all?(chunks, fn chunk ->
               # Chunks should contain actual text, not empty strings
               chunk.text != ""
             end)
    end

    test "handles single token text", %{tokenizer: tokenizer} do
      text = "Hello"
      {:ok, chunks} = Token.chunk(text, tokenizer)

      assert length(chunks) == 1
      [chunk] = chunks
      assert chunk.token_count == 1
      assert chunk.text == "Hello"
    end

    test "handles short text within single chunk", %{tokenizer: tokenizer} do
      text = "Hello, how are you?"
      {:ok, chunks} = Token.chunk(text, tokenizer)

      assert length(chunks) == 1
      [chunk] = chunks
      assert chunk.text == "Hello, how are you?"
    end

    test "correctly maps chunk indices to original text", %{tokenizer: tokenizer} do
      {:ok, chunks} = Token.chunk(@sample_text, tokenizer)

      Enum.each(chunks, fn chunk ->
        extracted_text =
          String.slice(@sample_text, chunk.start_index, chunk.end_index - chunk.start_index)

        assert String.trim(chunk.text) == String.trim(extracted_text)
      end)
    end

    test "handles complex markdown text", %{tokenizer: tokenizer} do
      {:ok, chunks} = Token.chunk(@complex_markdown, tokenizer)

      # Verify chunks are created and maintain markdown structure
      assert length(chunks) > 0

      # Verify indices map correctly
      Enum.each(chunks, fn chunk ->
        extracted_text =
          String.slice(@complex_markdown, chunk.start_index, chunk.end_index - chunk.start_index)

        assert String.trim(chunk.text) == String.trim(extracted_text)
      end)
    end

    test "maintains semantic coherence", %{tokenizer: tokenizer} do
      text = String.duplicate("Complete sentence one. Complete sentence two. ", 50)
      {:ok, chunks} = Token.chunk(text, tokenizer)

      # Check that most chunks end with complete sentences
      chunks_ending_with_period =
        chunks
        |> Enum.filter(fn chunk ->
          String.trim(chunk.text) != "" and chunk != List.last(chunks)
        end)
        |> Enum.count(fn chunk ->
          String.trim_trailing(chunk.text) |> String.ends_with?(".")
        end)

      # Assert that at least 75% of non-empty, non-last chunks end with periods
      # Exclude last chunk
      total_relevant_chunks = Enum.count(chunks) - 1
      assert chunks_ending_with_period / total_relevant_chunks > 0.75
    end

    test "handles unicode text correctly", %{tokenizer: tokenizer} do
      text = "Hello 👋 World! This is a test with émojis 🌍 and áccents."
      {:ok, chunks} = Token.chunk(text, tokenizer, chunk_size: 5, chunk_overlap: 1)

      assert length(chunks) > 0
      # Verify that unicode characters are preserved
      reconstructed =
        chunks
        |> Enum.map(& &1.text)
        |> Enum.join("")
        |> String.trim()

      assert reconstructed =~ "👋"
      assert reconstructed =~ "🌍"
      assert reconstructed =~ "é"
      assert reconstructed =~ "á"
    end

    test "maintains chunk size constraints with long tokens", %{tokenizer: tokenizer} do
      # Text with very long words that might challenge the chunking
      text =
        "supercalifragilisticexpialidocious antidisestablishmentarianism pneumonoultramicroscopicsilicovolcanoconiosis"

      chunk_size = 5
      {:ok, chunks} = Token.chunk(text, tokenizer, chunk_size: chunk_size, chunk_overlap: 1)

      assert Enum.all?(chunks, fn chunk ->
               chunk.token_count <= chunk_size
             end)
    end
  end

  # Chonkie python output was:
  #  Chunk 1:
  # Text:
  # The tiny hippo mascot of the Chonkie library brings a playful touch to text chunking
  # Tokens: 20
  # Start Index: 0
  # End Index: 85
  #
  # Chunk 2:
  # Text:  touch to text chunking.
  # It efficiently breaks down large texts into manageable pieces while maintaining context
  #
  # Tokens: 20
  # Start Index: 62
  # End Index: 175
  #
  # Chunk 3:
  # Text:  pieces while maintaining context
  # and meaning. Whether you're working with documents, articles, or any lengthy
  # Tokens: 20
  # Start Index: 141
  # End Index: 251
  #
  # Chunk 4:
  # Text:  articles, or any lengthy text,
  # Chonkie makes the process simple and reliable.
  #
  # Tokens: 19
  # Start Index: 226
  # End Index: 305
  #
  # Chunk 5:
  # Text:  and reliable.
  #
  # Tokens: 4
  # Start Index: 290
  # End Index: 305

  test "with gpt2 tokenizer it exactly matches Chonkie output" do
    text =
      "\n" <>
        """
        The tiny hippo mascot of the Chonkie library brings a playful touch to text chunking.
        It efficiently breaks down large texts into manageable pieces while maintaining context
        and meaning. Whether you're working with documents, articles, or any lengthy text,
        Chonkie makes the process simple and reliable.
        """

    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("gpt2")
    {:ok, chunks} = Token.chunk(text, tokenizer, chunk_size: 20, chunk_overlap: 5)

    assert [
             %Chunk{
               text:
                 "\nThe tiny hippo mascot of the Chonkie library brings a playful touch to text chunking",
               start_index: 0,
               end_index: 85,
               token_count: 20
             },
             %Chunk{
               text:
                 " touch to text chunking.\nIt efficiently breaks down large texts into manageable pieces while maintaining context\n",
               start_index: 62,
               end_index: 175,
               token_count: 20
             },
             %Chunk{
               text:
                 " pieces while maintaining context\nand meaning. Whether you're working with documents, articles, or any lengthy",
               start_index: 141,
               end_index: 251,
               token_count: 20
             },
             %Chunk{
               text:
                 " articles, or any lengthy text,\nChonkie makes the process simple and reliable.\n",
               start_index: 226,
               end_index: 305,
               token_count: 19
             },
             %Chunk{
               text: " and reliable.\n",
               start_index: 290,
               end_index: 305,
               token_count: 4
             }
           ] == chunks
  end
end
