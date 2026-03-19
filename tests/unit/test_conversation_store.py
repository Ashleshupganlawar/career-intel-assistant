from job_intel.storage import ConversationStore


def test_conversation_persistence(tmp_path):
    path = tmp_path / "conversations.json"
    store = ConversationStore(str(path))

    tid = store.create_thread("Test")
    store.update_context(tid, {"resume_text": "hello world"})
    store.add_message(tid, "user", "question", "question")
    store.add_message(tid, "assistant", "answer", "answer")

    store2 = ConversationStore(str(path))
    thread = store2.get_thread(tid)

    assert thread["context"]["resume_text"] == "hello world"
    assert len(thread["messages"]) == 2
    assert store2.get_short_term(tid)
    assert store2.get_long_term(tid)
