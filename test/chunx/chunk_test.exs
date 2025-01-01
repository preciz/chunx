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

    test "raises when token_count is zero or negative" do
      assert_raise FunctionClauseError, fn ->
        Chunk.new("text", 0, 10, 0)
      end

      assert_raise FunctionClauseError, fn ->
        Chunk.new("text", 0, 10, -1)
      end
    end

    test "raises when text is not a binary" do
      assert_raise FunctionClauseError, fn ->
        Chunk.new(:not_a_string, 0, 10, 1)
      end
    end
  end
end
