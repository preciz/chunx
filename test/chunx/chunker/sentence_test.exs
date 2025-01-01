defmodule Chunx.Chunker.SentenceTest do
  use ExUnit.Case, async: true
  doctest Chunx.Chunker.Sentence

  alias Chunx.Chunker.Sentence

  setup do
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("gpt2")
    {:ok, tokenizer: tokenizer}
  end

  @sample_text """
  The process of text chunking in RAG applications represents a delicate balance between competing requirements. On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context that can be understood and processed independently. On the other, we must optimize for information density, ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy. In this post, we explore the challenges of text chunking in RAG applications and propose a novel approach that leverages recent advances in transformer-based language models to achieve a more effective balance between these competing requirements.
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
      assert {:ok, []} = Chunx.Chunker.Sentence.chunk("", tokenizer)
    end

    test "handles single sentence", %{tokenizer: tokenizer} do
      text = "This is a single sentence."
      {:ok, [sentence_chunk]} = Chunx.Chunker.Sentence.chunk(text, tokenizer)

      assert sentence_chunk.text == "This is a single sentence."
      assert length(sentence_chunk.sentences) == 1
    end

    test "handles short text within single chunk", %{tokenizer: tokenizer} do
      text = "Hello, how are you? I am doing well."
      {:ok, [sentence_chunk]} = Chunx.Chunker.Sentence.chunk(text, tokenizer)

      assert sentence_chunk.text == "Hello, how are you? I am doing well."
      assert length(sentence_chunk.sentences) == 2
    end

    test "handles overlap correctly", %{tokenizer: tokenizer} do
      {:ok, chunks} =
        Chunx.Chunker.Sentence.chunk(@sample_text, tokenizer, chunk_size: 512, chunk_overlap: 128)

      # Check that consecutive chunks have overlapping content
      chunks
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.each(fn [chunk1, chunk2] ->
        assert chunk2.start_byte < chunk1.end_byte
      end)
    end

    test "ensures correct overlap token count between chunks", %{tokenizer: tokenizer} do
      desired_overlap = 10
      text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."

      {:ok, chunks} =
        Sentence.chunk(text, tokenizer, chunk_size: 20, chunk_overlap: desired_overlap)

      # Test consecutive chunks
      chunks
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.each(fn [chunk1, chunk2] ->
        overlapping_sentences =
          Enum.filter(chunk2.sentences, fn s ->
            s.start_byte < chunk1.end_byte
          end)

        overlap_tokens = Enum.sum_by(overlapping_sentences, & &1.token_count)

        assert overlap_tokens >= desired_overlap,
               "Expected overlap of #{desired_overlap} tokens, but got #{overlap_tokens}"
      end)
    end

    test "combines short sentences correctly", %{tokenizer: tokenizer} do
      text = "Hi! I am. Ok now. This is a longer sentence here. Very short! No! Testing."
      threshold = 10

      {:ok, chunks} =
        Sentence.chunk(text, tokenizer,
          chunk_size: 512,
          short_sentence_threshold: threshold
        )

      # The first three short sentences should be combined
      [first_chunk | _] = chunks
      first_sentences = first_chunk.sentences

      # Check that very short sentences were combined
      assert Enum.any?(first_sentences, fn s ->
               String.contains?(s.text, "Hi! I am. Ok now")
             end)

      # Verify longer sentences weren't combined
      assert Enum.any?(first_sentences, fn s ->
               s.text == " This is a longer sentence here."
             end)

      # Check that consecutive short sentences at the end were combined
      assert Enum.any?(first_sentences, fn s ->
               String.contains?(s.text, "Very short! No! Testing")
             end)
    end

    test "respects short_sentence_threshold configuration", %{tokenizer: tokenizer} do
      text = "A. B. Long sentence here. C. D."

      {:ok, chunks_with_high_threshold} =
        Sentence.chunk(text, tokenizer, short_sentence_threshold: 20)

      {:ok, chunks_with_low_threshold} =
        Sentence.chunk(text, tokenizer, short_sentence_threshold: 2)

      # With high threshold, A and B should be combined
      high_first_chunk = hd(chunks_with_high_threshold)

      assert Enum.any?(high_first_chunk.sentences, fn s ->
               String.contains?(s.text, "A. B.")
             end)

      # With low threshold, A and B should remain separate
      low_first_chunk = hd(chunks_with_low_threshold)

      assert Enum.any?(low_first_chunk.sentences, fn s ->
               s.text == "A."
             end)
    end

    test "respects minimum sentences per chunk", %{tokenizer: tokenizer} do
      text = "First sentence. Second sentence. Third sentence."
      min_sentences = 2

      {:ok, chunks} =
        Chunx.Chunker.Sentence.chunk(text, tokenizer,
          min_sentences_per_chunk: min_sentences,
          chunk_size: 3,
          chunk_overlap: 1
        )

      Enum.each(chunks, fn chunk ->
        sentence_count = chunk.text |> String.split(".") |> Enum.count(&(String.trim(&1) != ""))
        # Last chunk might have fewer sentences
        assert sentence_count >= min_sentences or chunk == List.last(chunks)
      end)
    end

    test "correctly maps chunk indices to original text", %{tokenizer: tokenizer} do
      {:ok, chunks} = Chunx.Chunker.Sentence.chunk(@sample_text, tokenizer)

      Enum.each(chunks, fn chunk ->
        extracted_text =
          binary_part(@sample_text, chunk.start_byte, chunk.end_byte - chunk.start_byte)

        assert chunk.text == extracted_text
      end)
    end

    test "correctly maps chunk indices in complex markdown", %{tokenizer: tokenizer} do
      {:ok, chunks} = Chunx.Chunker.Sentence.chunk(@complex_markdown, tokenizer)

      Enum.each(chunks, fn chunk ->
        extracted_text =
          binary_part(@complex_markdown, chunk.start_byte, chunk.end_byte - chunk.start_byte)

        assert chunk.text == extracted_text
      end)
    end

    test "validates chunk_size", %{tokenizer: tokenizer} do
      assert_raise ArgumentError, "chunk_size must be positive", fn ->
        Chunx.Chunker.Sentence.chunk("test", tokenizer, chunk_size: 0)
      end
    end

    test "validates min_sentences_per_chunk", %{tokenizer: tokenizer} do
      assert_raise ArgumentError, "min_sentences_per_chunk must be at least 1", fn ->
        Chunx.Chunker.Sentence.chunk("test", tokenizer, min_sentences_per_chunk: 0)
      end
    end

    test "respects chunk size limits", %{tokenizer: tokenizer} do
      chunk_size = 512
      {:ok, chunks} = Sentence.chunk(@sample_text, tokenizer, chunk_size: chunk_size)

      assert Enum.all?(chunks, fn chunk ->
               chunk.token_count <= chunk_size
             end)
    end

    test "handles custom delimiters", %{tokenizer: tokenizer} do
      text = "First segment<SEP>Second segment<SEP>Third segment<SEP>Fourth segment"

      {:ok, chunks} =
        Chunx.Chunker.Sentence.chunk(
          text,
          tokenizer,
          chunk_size: 30,
          chunk_overlap: 0,
          delimiters: ["<SEP>"]
        )

      assert length(chunks) > 0

      assert Enum.all?(chunks, fn chunk ->
               String.contains?(chunk.text, "segment")
             end)

      # Verify the original text is properly reconstructed from chunks
      reconstructed =
        chunks
        |> Enum.map(& &1.text)
        |> Enum.join("<SEP>")

      assert reconstructed == text
    end

    test "returns correct value for text", %{tokenizer: tokenizer} do
      {:ok, chunks} =
        Chunx.Chunker.Sentence.chunk(
          """
          Hi! How are you? I am fine. This is a short sentence. And another one!
          Yet another short one. These are all brief. Very brief indeed. Testing multiple sentences.
          Can you see how they group? They should fit several per chunk now!
          """,
          tokenizer,
          chunk_size: 30,
          chunk_overlap: 10,
          min_sentences_per_chunk: 1,
          short_sentence_threshold: 12
        )

      assert chunks == [
               %Chunx.SentenceChunk{
                 text:
                   "Hi! How are you? I am fine. This is a short sentence. And another one!\nYet another short one.",
                 sentences: [
                   %Chunx.Chunk{
                     text: "Hi!",
                     start_byte: 0,
                     end_byte: 3,
                     token_count: 2,
                     embedding: nil
                   },
                   %Chunx.Chunk{
                     text: " How are you? I am fine.",
                     start_byte: 3,
                     end_byte: 27,
                     token_count: 8,
                     embedding: nil
                   },
                   %Chunx.Chunk{
                     text: " This is a short sentence.",
                     start_byte: 27,
                     end_byte: 53,
                     token_count: 6,
                     embedding: nil
                   },
                   %Chunx.Chunk{
                     text: " And another one!",
                     start_byte: 53,
                     end_byte: 70,
                     token_count: 4,
                     embedding: nil
                   },
                   %Chunx.Chunk{
                     text: "\nYet another short one.",
                     start_byte: 70,
                     end_byte: 93,
                     token_count: 6,
                     embedding: nil
                   }
                 ],
                 start_byte: 0,
                 end_byte: 93,
                 token_count: 26
               },
               %Chunx.SentenceChunk{
                 text:
                   " And another one!\nYet another short one. These are all brief. Very brief indeed. Testing multiple sentences.",
                 sentences: [
                   %Chunx.Chunk{
                     text: " And another one!",
                     start_byte: 53,
                     end_byte: 70,
                     token_count: 4,
                     embedding: nil
                   },
                   %Chunx.Chunk{
                     text: "\nYet another short one.",
                     start_byte: 70,
                     end_byte: 93,
                     token_count: 6,
                     embedding: nil
                   },
                   %Chunx.Chunk{
                     text: " These are all brief.",
                     start_byte: 93,
                     end_byte: 114,
                     token_count: 5,
                     embedding: nil
                   },
                   %Chunx.Chunk{
                     text: " Very brief indeed.",
                     start_byte: 114,
                     end_byte: 133,
                     token_count: 4,
                     embedding: nil
                   },
                   %Chunx.Chunk{
                     text: " Testing multiple sentences.",
                     start_byte: 133,
                     end_byte: 161,
                     token_count: 4,
                     embedding: nil
                   }
                 ],
                 start_byte: 53,
                 end_byte: 161,
                 token_count: 23
               },
               %Chunx.SentenceChunk{
                 text:
                   " Very brief indeed. Testing multiple sentences.\nCan you see how they group? They should fit several per chunk now!\n",
                 sentences: [
                   %Chunx.Chunk{
                     text: " Very brief indeed.",
                     start_byte: 114,
                     end_byte: 133,
                     token_count: 4,
                     embedding: nil
                   },
                   %Chunx.Chunk{
                     text: " Testing multiple sentences.",
                     start_byte: 133,
                     end_byte: 161,
                     token_count: 4,
                     embedding: nil
                   },
                   %Chunx.Chunk{
                     text: "\nCan you see how they group?",
                     start_byte: 161,
                     end_byte: 189,
                     token_count: 8,
                     embedding: nil
                   },
                   %Chunx.Chunk{
                     text: " They should fit several per chunk now!\n",
                     start_byte: 189,
                     end_byte: 229,
                     token_count: 9,
                     embedding: nil
                   }
                 ],
                 start_byte: 114,
                 end_byte: 229,
                 token_count: 25
               }
             ]
    end
  end
end
