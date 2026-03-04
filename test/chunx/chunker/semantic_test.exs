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

  setup do
    {:ok, dynamic_supervisor} = DynamicSupervisor.start_link([])

    model = "intfloat/e5-small-v2"
    {:ok, model_info} = Bumblebee.load_model({:hf, model})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, model})

    serving =
      Bumblebee.Text.text_embedding(
        model_info,
        tokenizer,
        compile: [batch_size: 20, sequence_length: 512],
        defn_options: [compiler: EXLA]
      )
      |> Nx.Serving.defn_options(compiler: EXLA)

    {:ok, _} =
      DynamicSupervisor.start_child(
        dynamic_supervisor,
        {Nx.Serving,
         serving: serving, name: :embedding_serving, batch_size: 20, batch_timeout: 1000}
      )

    %{
      tokenizer: tokenizer.native_tokenizer,
      serving_fun: fn inputs ->
        Nx.Serving.batched_run(:embedding_serving, inputs) |> Enum.map(& &1.embedding)
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

      assert length(chunks) > 0
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

    test "uses auto threshold with small chunk_size", context do
      %{tokenizer: tokenizer, serving_fun: serving_fun} = context

      {:ok, chunks} =
        Semantic.chunk(
          @sample_text,
          tokenizer,
          serving_fun,
          threshold: :auto,
          chunk_size: 15,
          min_chunk_size: 1,
          threshold_step: 0.1
        )

      assert length(chunks) > 0
    end

    test "uses auto threshold with large chunk_size", context do
      %{tokenizer: tokenizer, serving_fun: serving_fun} = context

      {:ok, chunks} =
        Semantic.chunk(
          @sample_text,
          tokenizer,
          serving_fun,
          threshold: :auto,
          chunk_size: 5000,
          min_chunk_size: 1000,
          threshold_step: 0.1
        )

      assert length(chunks) > 0
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

    test "uses auto threshold forcing too large chunks", context do
      %{tokenizer: tokenizer, serving_fun: serving_fun} = context

      {:ok, _chunks} =
        Semantic.chunk(
          @sample_text,
          tokenizer,
          serving_fun,
          threshold: :auto,
          chunk_size: 1,
          min_chunk_size: 1,
          threshold_step: 0.2
        )
    end

    test "uses auto threshold with identical sentences", context do
      %{tokenizer: tokenizer, serving_fun: serving_fun} = context

      # Repeating the exact same sentence means similarities will have std = 0.0
      # low = median, high = median, high - low = 0.0 <= threshold_step
      {:ok, _chunks} =
        Semantic.chunk(
          "This is a test. This is a test. This is a test. This is a test.",
          tokenizer,
          serving_fun,
          threshold: :auto,
          threshold_step: 0.1
        )
    end

    test "uses auto threshold forcing too small chunks", context do
      %{tokenizer: tokenizer, serving_fun: serving_fun} = context

      {:ok, _chunks} =
        Semantic.chunk(
          @sample_text,
          tokenizer,
          serving_fun,
          threshold: :auto,
          chunk_size: 5000,
          min_chunk_size: 1000,
          threshold_step: 0.0001
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

      assert length(chunks) > 0
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
