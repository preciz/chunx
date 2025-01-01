defmodule Chunx.Chunker.WordTest do
  use ExUnit.Case, async: true
  doctest Chunx.Chunker.Word
  alias Chunx.Chunker.Word
  alias Chunx.Chunk

  setup do
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("gpt2")
    {:ok, tokenizer: tokenizer}
  end

  @sample_text """
  The process of text chunking in RAG applications represents a delicate balance between competing requirements.
  On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context
  that can be understood and processed independently. On the other, we must optimize for information density,
  ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy.
  In this post, we explore the challenges of text chunking in RAG applications and propose a novel approach that
  leverages recent advances in transformer-based language models to achieve a more effective balance between
  these competing requirements.
  """

  @complex_markdown """
  # Heading 1
  This is a paragraph with some **bold text** and _italic text_.
  ## Heading 2
  - Bullet point 1
  - Bullet point 2 with `inline code`
  ```python
  # Code block
  def hello_world():
      print("Hello, world!")
  ```
  Another paragraph with [a link](https://example.com) and an image:
  ![Alt text](https://example.com/image.jpg)
  > A blockquote with multiple lines
  > that spans more than one line.
  Finally, a paragraph at the end.
  """

  describe "chunk/3" do
    test "handles empty text", %{tokenizer: tokenizer} do
      assert {:ok, []} = Word.chunk("", tokenizer)
    end

    test "creates chunks with default options", %{tokenizer: tokenizer} do
      {:ok, chunks} = Word.chunk(@sample_text, tokenizer)

      assert length(chunks) > 0

      assert Enum.all?(chunks, fn chunk ->
               is_binary(chunk.text) and
                 chunk.token_count > 0 and
                 chunk.start_byte >= 0 and
                 chunk.end_byte > chunk.start_byte
             end)
    end

    test "respects chunk size", %{tokenizer: tokenizer} do
      {:ok, chunks} = Word.chunk(@sample_text, tokenizer, chunk_size: 50)

      assert Enum.all?(chunks, fn chunk ->
               chunk.token_count <= 50
             end)
    end

    test "handles single word text", %{tokenizer: tokenizer} do
      {:ok, chunks} = Word.chunk("Hello", tokenizer)

      assert length(chunks) == 1
      [chunk] = chunks
      assert chunk.token_count == 1
      assert chunk.text == "Hello"
    end

    test "handles short text within single chunk", %{tokenizer: tokenizer} do
      text = "Hello, how are you?"
      {:ok, chunks} = Word.chunk(text, tokenizer)

      assert length(chunks) == 1
      [chunk] = chunks
      assert chunk.text == "Hello, how are you?"
    end

    test "handles overlap correctly", %{tokenizer: tokenizer} do
      {:ok, chunks} = Word.chunk(@sample_text, tokenizer, chunk_size: 50, chunk_overlap: 10)

      # Check that consecutive chunks have overlapping content
      chunks
      # Only get pairs, discard incomplete chunks
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.each(fn [chunk1, chunk2] ->
        assert chunk2.start_byte < chunk1.end_byte,
               "Expected overlap between chunks not found"
      end)

      # Ensure we have at least 2 chunks for a meaningful overlap test
      assert length(chunks) >= 2, "Expected multiple chunks for overlap test"
    end

    test "validates chunk_size", %{tokenizer: tokenizer} do
      assert_raise ArgumentError, "chunk_size must be positive", fn ->
        Word.chunk("test", tokenizer, chunk_size: 0)
      end
    end

    test "validates overlap", %{tokenizer: tokenizer} do
      assert_raise ArgumentError,
                   "chunk_overlap must be less than chunk_size and non-negative",
                   fn ->
                     Word.chunk("test", tokenizer, chunk_size: 10, chunk_overlap: 10)
                   end
    end

    test "correctly maps chunk indices to original text", %{tokenizer: tokenizer} do
      {:ok, chunks} = Word.chunk(@sample_text, tokenizer)

      Enum.each(chunks, fn chunk ->
        extracted_text =
          String.slice(@sample_text, chunk.start_byte, chunk.end_byte - chunk.start_byte)

        assert String.trim(chunk.text) == String.trim(extracted_text)
      end)
    end

    test "handles complex markdown text", %{tokenizer: tokenizer} do
      {:ok, chunks} = Word.chunk(@complex_markdown, tokenizer)

      # Verify chunks are created and maintain markdown structure
      assert length(chunks) > 0

      # Verify indices map correctly
      Enum.each(chunks, fn chunk ->
        extracted_text =
          String.slice(@complex_markdown, chunk.start_byte, chunk.end_byte - chunk.start_byte)

        assert String.trim(chunk.text) == String.trim(extracted_text)
      end)
    end
  end

  # Chonkie python output was:
  # Chunk 1:
  # Text: '
  # The WordChunker splits text at word boundaries, which'
  # Tokens: 12
  # Start Index: 0
  # End Index: 54
  #
  # Chunk 2:
  # Text: ' boundaries, which makes it ideal for
  # preserving complete'
  # Tokens: 11
  # Start Index: 36
  # End Index: 93
  #
  # Chunk 3:
  # Text: ' complete words. It won't cut words in the middle,'
  # Tokens: 12
  # Start Index: 84
  # End Index: 134
  #
  # Chunk 4:
  # Text: ' the middle, making the output more
  # readable and semantically'
  # Tokens: 12
  # Start Index: 122
  # End Index: 183
  #
  # Chunk 5:
  # Text: ' and semantically coherent. This is especially useful when working with'
  # Tokens: 12
  # Start Index: 166
  # End Index: 237
  #
  # Chunk 6:
  # Text: ' when working with
  # natural language processing tasks.
  # '
  # Tokens: 10
  # Start Index: 219
  # End Index: 273
  test "with gpt2 tokenizer it creates expected chunks", %{tokenizer: tokenizer} do
    text =
      "\n" <>
        """
        The WordChunker splits text at word boundaries, which makes it ideal for
        preserving complete words. It won't cut words in the middle, making the output more
        readable and semantically coherent. This is especially useful when working with
        natural language processing tasks.
        """

    {:ok, chunks} = Word.chunk(text, tokenizer, chunk_size: 12, chunk_overlap: 3)

    assert [
             %Chunk{
               text: "\nThe WordChunker splits text at word boundaries, which",
               start_byte: 0,
               end_byte: 54,
               token_count: 12
             },
             %Chunk{
               text: " boundaries, which makes it ideal for\npreserving complete",
               start_byte: 36,
               end_byte: 93,
               token_count: 11
             },
             %Chunk{
               text: " complete words. It won't cut words in the middle,",
               start_byte: 84,
               end_byte: 134,
               token_count: 12
             },
             %Chunk{
               text: " the middle, making the output more\nreadable and semantically",
               start_byte: 122,
               end_byte: 183,
               token_count: 12
             },
             %Chunk{
               text: " and semantically coherent. This is especially useful when working with",
               start_byte: 166,
               end_byte: 237,
               token_count: 12
             },
             %Chunk{
               text: " when working with\nnatural language processing tasks.\n",
               start_byte: 219,
               end_byte: 273,
               token_count: 10
             }
           ] == chunks
  end

  # Chonkie python output was:
  # Chunk 1:
  # Text: 'Hey there my'
  # Tokens: 3
  # Start Index: 0
  # End Index: 12
  #
  # Chunk 2:
  # Text: ' friend, how'
  # Tokens: 3
  # Start Index: 12
  # End Index: 24
  #
  # Chunk 3:
  # Text: ' is it going'
  # Tokens: 3
  # Start Index: 24
  # End Index: 36
  #
  # Chunk 4:
  # Text: ' out there?'
  # Tokens: 3
  # Start Index: 36
  # End Index: 47
  test "short string with small chunk size", %{tokenizer: tokenizer} do
    text = "Hey there my friend, how is it going out there?"

    {:ok, chunks} = Word.chunk(text, tokenizer, chunk_size: 3, chunk_overlap: 0.25)

    assert [
             %Chunk{text: "Hey there my", start_byte: 0, end_byte: 12, token_count: 3},
             %Chunk{text: " friend, how", start_byte: 12, end_byte: 24, token_count: 3},
             %Chunk{text: " is it going", start_byte: 24, end_byte: 36, token_count: 3},
             %Chunk{text: " out there?", start_byte: 36, end_byte: 47, token_count: 3}
           ] == chunks
  end
end
