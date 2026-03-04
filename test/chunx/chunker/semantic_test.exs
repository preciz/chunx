defmodule Chunx.Chunker.SemanticTest do
  use ExUnit.Case, async: true

  alias Chunx.Chunker.Semantic
  alias Chunx.SentenceChunk

  @sample_text """
  The process of text chunking in RAG applications represents a delicate balance between competing requirements. On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context that can be understood and processed independently. On the other, we must optimize for information density, ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy. In this post, we explore the challenges of text chunking in RAG applications and propose a novel approach that leverages recent advances in transformer-based language models to achieve a more effective balance between these competing requirements.
  """

  @sample_complex_markdown """
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

  setup_all do
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("gpt2")

    %{
      tokenizer: tokenizer,
      serving_fun: fn inputs ->
        # Use a mock embedding function that returns a random tensor 
        # so that standard deviations and medians are still calculable for :auto threshold
        Enum.map(inputs, fn _ ->
          Nx.tensor(Enum.map(1..768, fn _ -> :rand.uniform() end))
        end)
      end
    }
  end

  describe "chunk/4" do
    test "initializes with default parameters", context do
      %{tokenizer: tokenizer, serving_fun: serving_fun} = context

      {:ok, chunks} =
        Semantic.chunk(
          @sample_text,
          tokenizer,
          serving_fun
        )

      assert chunks != []
      assert Enum.all?(chunks, &match?(%SentenceChunk{}, &1))
    end

    test "handles empty text", context do
      %{tokenizer: tokenizer, serving_fun: serving_fun} = context

      {:ok, chunks} =
        Semantic.chunk(
          "",
          tokenizer,
          serving_fun
        )

      assert chunks == []
    end

    test "handles single sentence", context do
      text = "This is a single sentence."
      %{tokenizer: tokenizer, serving_fun: serving_fun} = context

      {:ok, chunks} =
        Semantic.chunk(
          text,
          tokenizer,
          serving_fun
        )

      assert length(chunks) == 1
      assert hd(chunks).text == text
    end

    test "uses auto threshold forcing too large chunks", context do
      %{tokenizer: tokenizer, serving_fun: serving_fun} = context

      # We want initial threshold to result in chunks that are larger than chunk_size.
      # If we set chunk_size very low, it will find the initial chunks too large and increase threshold.
      {:ok, _chunks} =
        Semantic.chunk(
          @sample_text,
          tokenizer,
          serving_fun,
          threshold: :auto,
          # This forces the chunks to be considered "too large" initially
          chunk_size: 1,
          min_chunk_size: 1,
          threshold_step: 0.0001
        )
    end

    test "uses auto threshold forcing too small chunks", context do
      %{tokenizer: tokenizer, serving_fun: serving_fun} = context

      # We want initial threshold to result in chunks that are smaller than min_chunk_size.
      {:ok, _chunks} =
        Semantic.chunk(
          @sample_text,
          tokenizer,
          serving_fun,
          threshold: :auto,
          chunk_size: 5000,
          # This forces the chunks to be considered "too small" initially
          min_chunk_size: 1000,
          threshold_step: 0.0001
        )
    end

    test "forces min_sentences even if exceeding chunk_size", context do
      %{tokenizer: tokenizer, serving_fun: serving_fun} = context

      {:ok, _chunks} =
        Semantic.chunk(
          "First long sentence that has many tokens. Second long sentence that also has many tokens. Third long sentence with even more tokens.",
          tokenizer,
          serving_fun,
          chunk_size: 2,
          min_sentences: 2
        )
    end

    test "validates all configuration arguments", context do
      %{tokenizer: tokenizer, serving_fun: serving_fun} = context

      assert_raise ArgumentError, "chunk_size must be positive", fn ->
        Semantic.chunk(@sample_text, tokenizer, serving_fun, chunk_size: 0)
      end

      assert_raise ArgumentError, "min_sentences must be positive", fn ->
        Semantic.chunk(@sample_text, tokenizer, serving_fun, min_sentences: 0)
      end

      assert_raise ArgumentError, "min_chunk_size must be positive", fn ->
        Semantic.chunk(@sample_text, tokenizer, serving_fun, min_chunk_size: 0)
      end

      assert_raise ArgumentError, "threshold_step must be between 0 and 1", fn ->
        Semantic.chunk(@sample_text, tokenizer, serving_fun, threshold_step: 1.5)
      end

      assert_raise ArgumentError, "threshold must be :auto or a float between 0 and 1", fn ->
        Semantic.chunk(@sample_text, tokenizer, serving_fun, threshold: "invalid")
      end
    end

    test "splits groups when exceeding chunk_size and meeting min_sentences", context do
      %{tokenizer: tokenizer, serving_fun: serving_fun} = context

      {:ok, chunks} =
        Semantic.chunk(
          "Sentence one here. Sentence two here. Sentence three here. Sentence four here.",
          tokenizer,
          serving_fun,
          # Force all into one group
          threshold: 0.0,
          chunk_size: 6,
          min_sentences: 1
        )

      assert length(chunks) > 1
    end

    test "respects chunk size limits", context do
      %{tokenizer: tokenizer, serving_fun: serving_fun} = context

      {:ok, chunks} =
        Semantic.chunk(
          @sample_text,
          tokenizer,
          serving_fun,
          chunk_size: 512
        )

      assert Enum.all?(chunks, fn chunk -> chunk.token_count <= 512 end)
    end

    test "handles complex markdown text", context do
      %{tokenizer: tokenizer, serving_fun: serving_fun} = context

      {:ok, chunks} =
        Semantic.chunk(
          @sample_complex_markdown,
          tokenizer,
          serving_fun
        )

      assert chunks != []
      assert Enum.all?(chunks, &match?(%SentenceChunk{}, &1))
    end

    test "validates configuration", context do
      assert_raise ArgumentError, fn ->
        %{tokenizer: tokenizer, serving_fun: serving_fun} = context

        Semantic.chunk(
          @sample_text,
          tokenizer,
          serving_fun,
          chunk_size: 0
        )
      end
    end

    test "maintains correct text indices", context do
      %{tokenizer: tokenizer, serving_fun: serving_fun} = context

      {:ok, chunks} =
        Semantic.chunk(
          @sample_text,
          tokenizer,
          serving_fun
        )

      Enum.each(chunks, fn chunk ->
        extracted_text =
          binary_part(@sample_text, chunk.start_byte, chunk.end_byte - chunk.start_byte)

        assert String.trim(chunk.text) == String.trim(extracted_text)
      end)
    end

    test "different similarity thresholds affect chunk count", context do
      %{tokenizer: tokenizer, serving_fun: serving_fun} = context

      {:ok, chunks_high_threshold} =
        Semantic.chunk(
          @sample_text,
          tokenizer,
          serving_fun,
          threshold: 0.9
        )

      {:ok, chunks_low_threshold} =
        Semantic.chunk(
          @sample_text,
          tokenizer,
          serving_fun,
          threshold: 0.1
        )

      assert length(chunks_high_threshold) >= length(chunks_low_threshold)
    end
  end
end
