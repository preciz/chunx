defmodule Chunx.Chunker.PropertyTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Chunx.Chunker.{Token, Word, Sentence}

  setup_all do
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("gpt2")
    %{tokenizer: tokenizer}
  end

  property "Token chunker always produces chunks within the specified size", %{
    tokenizer: tokenizer
  } do
    check all(
            text <- string(:printable),
            chunk_size <- integer(1..100)
          ) do
      {:ok, chunks} = Token.chunk(text, tokenizer, chunk_size: chunk_size)

      for chunk <- chunks do
        assert chunk.token_count <= chunk_size
      end
    end
  end

  property "Word chunker always produces chunks within the specified size or minimum size for a single word",
           %{tokenizer: tokenizer} do
    check all(
            text <- string(:printable),
            chunk_size <- integer(1..100)
          ) do
      # Set chunk_overlap to 0 to prevent previous overlapping words from inflating the count of a single-word chunk
      {:ok, chunks} = Word.chunk(text, tokenizer, chunk_size: chunk_size, chunk_overlap: 0)

      for chunk <- chunks do
        words_in_chunk = Regex.scan(~r/\s*\S+/, chunk.text)
        assert chunk.token_count <= chunk_size or length(words_in_chunk) <= 1
      end
    end
  end

  property "Word chunker combines chunks to original text when overlap is 0", %{
    tokenizer: tokenizer
  } do
    check all(
            text <- string(:printable),
            chunk_size <- integer(1..50)
          ) do
      {:ok, chunks} = Word.chunk(text, tokenizer, chunk_size: chunk_size, chunk_overlap: 0)

      for chunk <- chunks do
        assert String.contains?(text, String.trim(chunk.text))

        extracted_text = binary_part(text, chunk.start_byte, chunk.end_byte - chunk.start_byte)
        assert chunk.text == extracted_text
      end
    end
  end

  property "Token chunker correctly tracks byte offsets", %{tokenizer: tokenizer} do
    check all(
            text <- string(:printable),
            chunk_size <- integer(1..50)
          ) do
      {:ok, chunks} = Token.chunk(text, tokenizer, chunk_size: chunk_size)

      for chunk <- chunks do
        extracted_text = binary_part(text, chunk.start_byte, chunk.end_byte - chunk.start_byte)
        assert chunk.text == extracted_text
      end
    end
  end

  property "Sentence chunker correctly tracks byte offsets", %{tokenizer: tokenizer} do
    check all(
            text <- string(:printable),
            chunk_size <- integer(10..100)
          ) do
      {:ok, chunks} = Sentence.chunk(text, tokenizer, chunk_size: chunk_size, chunk_overlap: 0)

      for chunk <- chunks do
        extracted_text = binary_part(text, chunk.start_byte, chunk.end_byte - chunk.start_byte)
        assert chunk.text == extracted_text
      end
    end
  end
end
