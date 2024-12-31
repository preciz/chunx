defmodule Chunx.SentenceChunkTest do
  use ExUnit.Case, async: true
  alias Chunx.{Chunk, SentenceChunk}

  describe "new/5" do
    test "creates a valid sentence chunk with proper parameters" do
      sentences = [%Chunk{text: "Test sentence.", start_index: 0, end_index: 13, token_count: 2}]
      chunk = SentenceChunk.new("sample text", 0, 10, 2, sentences)
      assert %SentenceChunk{} = chunk
      assert chunk.text == "sample text"
      assert chunk.start_index == 0
      assert chunk.end_index == 10
      assert chunk.token_count == 2
      assert chunk.sentences == sentences
    end

    test "raises when start_index is negative" do
      assert_raise FunctionClauseError, fn ->
        SentenceChunk.new("text", -1, 10, 1, [])
      end
    end

    test "raises when end_index is less than start_index" do
      assert_raise FunctionClauseError, fn ->
        SentenceChunk.new("text", 10, 5, 1, [])
      end
    end

    test "raises when token_count is zero or negative" do
      assert_raise FunctionClauseError, fn ->
        SentenceChunk.new("text", 0, 10, 0, [])
      end

      assert_raise FunctionClauseError, fn ->
        SentenceChunk.new("text", 0, 10, -1, [])
      end
    end

    test "raises when text is not a binary" do
      assert_raise FunctionClauseError, fn ->
        SentenceChunk.new(:not_a_string, 0, 10, 1, [])
      end
    end

    test "raises when sentences is not a list" do
      assert_raise FunctionClauseError, fn ->
        SentenceChunk.new("text", 0, 10, 1, "not a list")
      end
    end
  end
end
