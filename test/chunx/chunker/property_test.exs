defmodule Chunx.Chunker.PropertyTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Chunx.Chunker.{Sentence, Token, Word}
  alias Chunx.Tokenizer, as: TokenizerBoundary

  setup_all do
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("gpt2")
    %{tokenizer: tokenizer}
  end

  property "Token chunker respects size unless one byte span represents too many tokens", %{
    tokenizer: tokenizer
  } do
    check all(
            text <- string(:printable),
            chunk_size <- integer(1..100)
          ) do
      {:ok, chunks} =
        Token.chunk(text, tokenizer, chunk_size: chunk_size, chunk_overlap: 0)

      for chunk <- chunks do
        {:ok, offsets} = TokenizerBoundary.offsets(tokenizer, chunk.text)
        units = TokenizerBoundary.units(offsets)

        assert chunk.token_count <= chunk_size or
                 Enum.any?(units, fn {_, _, count} -> count > chunk_size end)
      end
    end
  end

  property "Word chunker always produces chunks within the specified size or minimum size for a single word",
           %{tokenizer: tokenizer} do
    check all(
            text <- string(:printable),
            chunk_size <- integer(1..100),
            chunk_overlap <- integer(0..(chunk_size - 1))
          ) do
      {:ok, chunks} =
        Word.chunk(text, tokenizer,
          chunk_size: chunk_size,
          chunk_overlap: chunk_overlap
        )

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
