from .models import LLM, EmbeddingModel


class Memory:
    """Simple memory module to store conversation history."""

    def __init__(self):
        self.messages = []

    def begin_turn(self) -> None:
        """Mark the start of an agent turn."""

    def end_turn(self) -> None:
        """Mark the end of an agent turn."""

    def clear(self) -> None:
        """Clear conversation history while keeping system messages."""
        self.messages = [
            message for message in self.messages if message["role"] == "system"
        ]

    def add(
        self, role: str, content: str, tool_call: dict | None = None, **kwargs
    ) -> None:
        """Add a message to memory."""
        message = {"role": role, "content": content}

        # Tool call
        if tool_call:
            message["tool_calls"] = [tool_call]

        # Append message to memory
        self.messages.append(message)

    def get_messages(self) -> list[dict]:
        """Get all messages."""
        return self.messages


class TrimmingMemory(Memory):
    """Memory that keeps the system prompt and the most recent complete turns."""

    def __init__(self, max_turns: int = 2):
        super().__init__()
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.max_turns = max_turns
        self._system_messages = []
        self._completed_turns = []
        self._current_turn = []

    def begin_turn(self) -> None:
        """Keep room for the incoming turn and retain only complete prior turns."""
        if self._current_turn:
            raise RuntimeError(
                "Cannot begin a new turn before ending the current turn."
            )
        if self.max_turns > 1:
            self._completed_turns = self._completed_turns[-(self.max_turns - 1) :]
        else:
            self._completed_turns = []
        self._sync_messages()

    def end_turn(self) -> None:
        """Commit the active turn, then discard only older complete turns."""
        if self._current_turn:
            self._completed_turns.append(self._current_turn)
            self._current_turn = []
            self._completed_turns = self._completed_turns[-self.max_turns :]
        self._sync_messages()

    def clear(self) -> None:
        """Clear completed and active turns while keeping system messages."""
        self._completed_turns = []
        self._current_turn = []
        self._sync_messages()

    def add(self, role: str, content: str, **kwargs) -> None:
        super().add(role, content, **kwargs)
        message = self.messages[-1]
        if role == "system":
            self._system_messages.append(message)
        else:
            self._current_turn.append(message)

    def _sync_messages(self) -> None:
        self.messages = [
            *self._system_messages,
            *(message for turn in self._completed_turns for message in turn),
            *self._current_turn,
        ]


class SummarizationMemory(Memory):
    """Memory that compresses past turns into a running summary via an LLM."""

    def __init__(self, llm: LLM):
        super().__init__()
        self.llm = llm

    def add(self, role: str, content: str, **kwargs) -> None:
        # Add the new message first using the parent class (Memory)
        super().add(role, content, **kwargs)

        # After each completed turn, update the running summary
        if role == "assistant":
            # Split the existing summary from the new conversation turns
            summary = ""
            conversation = ""
            for message in self.messages:
                if message["role"] == "system":
                    summary = message["content"]
                else:
                    conversation += f"{message['role']}: {message['content']}\n"

            # Ask the LLM to extend the summary with the new conversation
            prompt = f"""Update the summary with the new conversation.

Summary: {summary}

Conversation:
{conversation}
Output the updated summary only."""
            response = self.llm.generate([{"role": "user", "content": prompt}])
            self.messages = [{"role": "system", "content": response.content}]


class LongTermMemory(Memory):
    """Memory enhanced with RAG: augments user queries with retrieved documents."""

    def __init__(self, embedding_model: EmbeddingModel, documents: list[str]):
        super().__init__()
        self.embedding_model = embedding_model
        self.documents = documents
        self.embeddings = [embedding_model.embed(doc) for doc in documents]

    def add(self, role: str, content: str, **kwargs) -> None:
        # Augment user queries with retrieved context before storing
        if role == "user":
            context = "\n".join(self.search(content))
            content = f"""Context:
{context}

Question: {content}"""
        super().add(role, content, **kwargs)

    def search(self, query: str, k: int = 3) -> list[str]:
        """Return the top-k documents most similar to the query."""
        query_embedding = self.embedding_model.embed(query)
        scores = [self._cosine(query_embedding, emb) for emb in self.embeddings]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.documents[i] for i in ranked[:k]]

    def _cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
        return dot / norm


class HybridMemory(Memory):
    """Memory combining RAG, periodic summarization, and a recent turn window."""

    def __init__(
        self,
        llm: LLM,
        embedding_model: EmbeddingModel,
        documents: list[str],
        max_recent_turns: int = 3,
    ):
        super().__init__()
        self.llm = llm
        self.embedding_model = embedding_model
        self.documents = documents
        self.embeddings = [embedding_model.embed(doc) for doc in documents]
        self.max_recent_turns = max_recent_turns
        self.summary = ""
        self._raw_turns = []  # List of completed turns (each turn is a list of messages)
        self._current_turn = []

    def add(self, role: str, content: str, **kwargs) -> None:
        super().add(role, content, **kwargs)
        message = self.messages[-1]
        self._current_turn.append(message)

        # When an assistant finishes a turn, evaluate if we should chunk/summarize
        if role == "assistant":
            self._raw_turns.append(self._current_turn)
            self._current_turn = []

            # If we exceed our recent turn window, fold the oldest turn into the summary
            if len(self._raw_turns) > self.max_recent_turns:
                old_turn = self._raw_turns.pop(0)
                self._update_summary(old_turn)

    def _update_summary(self, turn_messages: list[dict]) -> None:
        turn_text = "\n".join([f"{m['role']}: {m['content']}" for m in turn_messages])
        prompt = f"""Extend the existing summary with the following new interaction turn. Keep it concise.

Existing Summary:
{self.summary if self.summary else "None"}

New Turn:
{turn_text}

Updated Summary:"""
        response = self.llm.generate([{"role": "user", "content": prompt}])
        self.summary = response.content

    def get_messages(self, current_query: str = "") -> list[dict]:
        """Dynamically build the payload with RAG context, summary, and recent turns."""
        compiled_messages = []

        # Inject RAG context dynamically if a query is provided
        if current_query:
            query_emb = self.embedding_model.embed(current_query)
            scores = [
                sum(a * b for a, b in zip(query_emb, emb))
                / (
                    (sum(x * x for x in query_emb) ** 0.5)
                    * (sum(y * y for y in emb) ** 0.5)
                )
                for emb in self.embeddings
            ]
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            retrieved = "\n".join([self.documents[i] for i in ranked[:2]])

            compiled_messages.append(
                {
                    "role": "system",
                    "content": f"Relevant background knowledge:\n{retrieved}",
                }
            )

        # Inject running summary if available
        if self.summary:
            compiled_messages.append(
                {
                    "role": "system",
                    "content": f"Summary of earlier conversation:\n{self.summary}",
                }
            )

        # Append recent raw turns and current active turn
        for turn in self._raw_turns:
            compiled_messages.extend(turn)
        compiled_messages.extend(self._current_turn)

        return compiled_messages
