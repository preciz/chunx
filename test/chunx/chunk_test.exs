defmodule Chunx.ChunkTest do
  use ExUnit.Case, async: true
  alias Chunx.Chunk

  describe "new/4" do
    test "creates a valid chunk with proper parameters" do
      chunk = Chunk.new("sample text", 0, 10, 2)
      assert %Chunk{} = chunk
      assert chunk.text == "sample text"
      assert chunk.start_byte == 0
      assert chunk.end_byte == 10
      assert chunk.token_count == 2
    end

    test "raises when start_byte is negative" do
      assert_raise FunctionClauseError, fn ->
        Chunk.new("text", -1, 10, 1)
      end
    end

    test "raises when end_byte is less than start_byte" do
      assert_raise FunctionClauseError, fn ->
        Chunk.new("text", 10, 5, 1)
      end
    end

    test "accepts zero token_count for tokenizer-ignored text" do
      assert %Chunk{token_count: 0} = Chunk.new(<<0>>, 0, 1, 0)
    end

    test "raises when token_count is negative" do
      assert_raise FunctionClauseError, fn ->
        Chunk.new("text", 0, 10, -1)
      end
    end

    test "raises when text is not a binary" do
      assert_raise FunctionClauseError, fn ->
        call_constructor([:not_a_string, 0, 10, 1])
      end
    end

    test "raises a clear error when text is not valid UTF-8" do
      assert_raise ArgumentError, "text must be valid UTF-8", fn ->
        Chunk.new(<<255>>, 0, 1, 1)
      end
    end
  end

  defp call_constructor(arguments), do: apply(Chunk, :new, arguments)
end
