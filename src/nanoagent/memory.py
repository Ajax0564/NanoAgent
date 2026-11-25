from .models import LLM, EmbeddingModel


class Memory:
    """Simple memory module to store conversation history."""

    def __init__(self):
        """Create an empty message history.

        Inputs:
            None.

        Returns:
            None. Initializes ``messages`` as an empty list.
        """
        self.messages = []

    def begin_turn(self) -> None:
        """Provide a turn-start hook for memory implementations.

        Inputs:
            None.

        Returns:
            None. The base implementation does not alter the history.
        """

    def end_turn(self) -> None:
        """Provide a turn-end hook for memory implementations.

        Inputs:
            None.

        Returns:
            None. The base implementation does not alter the history.
        """

    def clear(self) -> None:
        """Remove all non-system messages from the history.

        Inputs:
            None.

        Returns:
            None. Retains messages whose role is ``system``.
        """
        self.messages = [
            message for message in self.messages if message["role"] == "system"
        ]

    def add(
        self, role: str, content: str, tool_call: dict | None = None, **kwargs
    ) -> None:
        """Append a role/content message and optional tool call.

        Args:
            role: Message role, such as ``system``, ``user``, or ``assistant``.
            content: Text content stored with the message.
            tool_call: Optional call mapping stored in a one-element
                ``tool_calls`` list.
            **kwargs: Extra values accepted for subclass compatibility; the base
                implementation does not store them.

        Returns:
            None. Appends a message dictionary to ``messages``.
        """
        message = {"role": role, "content": content}

        # Tool call
        if tool_call:
            message["tool_calls"] = [tool_call]

        # Append message to memory
        self.messages.append(message)

    def get_messages(self) -> list[dict]:
        """Return the stored message list.

        Inputs:
            None.

        Returns:
            The internal list itself, in insertion order; callers can mutate it.
        """
        return self.messages


class TrimmingMemory(Memory):
    """Memory that keeps the system prompt and the most recent complete turns."""

    def __init__(self, max_turns: int = 2):
        """Initialize memory that retains a bounded window of complete turns.

        Args:
            max_turns: Maximum number of completed turns retained alongside the
                active turn. Must be at least one.

        Returns:
            None. Starts with no system messages or turns.

        Raises:
            ValueError: If ``max_turns`` is less than one.
        """
        super().__init__()
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.max_turns = max_turns
        self._system_messages = []
        self._completed_turns = []
        self._current_turn = []

    def begin_turn(self) -> None:
        """Prepare a fresh active turn while reserving one retention slot for it.

        Inputs:
            None.

        Returns:
            None. Trims old completed turns and rebuilds the visible message list.

        Raises:
            RuntimeError: If another turn is still active.
        """
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
        """Commit the active turn and enforce the completed-turn limit.

        Inputs:
            None.

        Returns:
            None. Empty active turns are not added to the completed history.
        """
        if self._current_turn:
            self._completed_turns.append(self._current_turn)
            self._current_turn = []
            self._completed_turns = self._completed_turns[-self.max_turns :]
        self._sync_messages()

    def clear(self) -> None:
        """Discard completed and active turns but preserve system messages.

        Inputs:
            None.

        Returns:
            None. Rebuilds the visible message list from retained system messages.
        """
        self._completed_turns = []
        self._current_turn = []
        self._sync_messages()

    def add(self, role: str, content: str, **kwargs) -> None:
        """Store a message and place it in system memory or the active turn.

        Args:
            role: Message role; ``system`` messages are retained separately.
            content: Message text.
            **kwargs: Optional message fields handled by ``Memory.add``.

        Returns:
            None. Updates both the base message list and the relevant turn list.
        """
        super().add(role, content, **kwargs)
        message = self.messages[-1]
        if role == "system":
            self._system_messages.append(message)
        else:
            self._current_turn.append(message)

    def _sync_messages(self) -> None:
        """Rebuild the visible list from systems, completed turns, and active turn.

        Inputs:
            None.

        Returns:
            None. Replaces ``messages`` with the current retained conversation.
        """
        self.messages = [
            *self._system_messages,
            *(message for turn in self._completed_turns for message in turn),
            *self._current_turn,
        ]


class SummarizationMemory(Memory):
    """Memory that compresses past turns into a running summary via an LLM."""

    def __init__(self, llm: LLM):
        """Initialize memory that replaces history with an LLM-written summary.

        Args:
            llm: Model client used to update the summary after assistant messages.

        Returns:
            None. The base conversation history starts empty.
        """
        super().__init__()
        self.llm = llm

    def add(self, role: str, content: str, **kwargs) -> None:
        """Append a message and summarize the conversation on assistant output.

        Args:
            role: Message role.
            content: Message text to append before summarization.
            **kwargs: Optional fields accepted by the base memory implementation.

        Returns:
            None. On an ``assistant`` message, replaces the history with the
            updated summary as a single system message.
        """
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
        """Initialize document memory and precompute all document embeddings.

        Args:
            embedding_model: Model used for query and document vectors.
            documents: Text snippets searched when user messages are added.

        Returns:
            None. Stores the documents and their embeddings in matching order.
        """
        super().__init__()
        self.embedding_model = embedding_model
        self.documents = documents
        self.embeddings = [embedding_model.embed(doc) for doc in documents]

    def add(self, role: str, content: str, **kwargs) -> None:
        """Add a message, augmenting user content with the top retrieved documents.

        Args:
            role: Message role.
            content: Message text; user text is treated as the retrieval query.
            **kwargs: Optional message fields passed to the base implementation.

        Returns:
            None. Stores non-user messages unchanged and prefixes user messages
            with up to three relevant document snippets.
        """
        # Augment user queries with retrieved context before storing
        if role == "user":
            context = "\n".join(self.search(content))
            content = f"""Context:
{context}

Question: {content}"""
        super().add(role, content, **kwargs)

    def search(self, query: str, k: int = 3) -> list[str]:
        """Rank stored documents by cosine similarity to a query embedding.

        Args:
            query: Text to embed and compare with the stored document vectors.
            k: Maximum number of highest-scoring documents to return.

        Returns:
            Up to ``k`` document strings, ordered from most to least similar.
        """
        query_embedding = self.embedding_model.embed(query)
        scores = [self._cosine(query_embedding, emb) for emb in self.embeddings]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.documents[i] for i in ranked[:k]]

    def _cosine(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two numeric vectors.

        Args:
            a: First vector.
            b: Second vector, expected to have the same dimensions as ``a``.

        Returns:
            Dot product divided by the product of the vector norms.

        Raises:
            ZeroDivisionError: If either vector has zero magnitude.
        """
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
        """Initialize hybrid retrieval, summarization, and recent-turn memory.

        Args:
            llm: Model used to summarize old conversation turns.
            embedding_model: Model used to embed documents and queries.
            documents: Background snippets made available through retrieval.
            max_recent_turns: Number of recent turns kept unsummarized.

        Returns:
            None. Precomputes document embeddings and initializes empty turn and
            summary state.
        """
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
        """Store a message and summarize turns that exceed the recent window.

        Args:
            role: Message role; an assistant message marks the end of a turn.
            content: Message text.
            **kwargs: Optional fields accepted by the base memory implementation.

        Returns:
            None. Moves the oldest completed turn into the running summary when
            the recent-turn limit is exceeded.
        """
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
        """Extend the running summary with one completed conversation turn.

        Args:
            turn_messages: Ordered messages from the turn being summarized.

        Returns:
            None. Replaces ``summary`` with the model's updated summary text.
        """
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
        """Assemble model messages from retrieval, summary, and recent turns.

        Args:
            current_query: Optional text used to retrieve the two most relevant
                background documents.

        Returns:
            A new list of system context messages followed by retained raw turns
            and the active turn.
        """
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
