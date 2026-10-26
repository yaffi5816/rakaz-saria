"""The agent loop: keep going while the model is still asking for tools.

Two rules carry the whole file.

The loop condition is stop_reason, not a count of tool blocks and not a guess
about whether the model sounded finished. A turn ends when the model says it has
ended.

All tool results go back in one user message. The model may ask for several
tools at once, and answering them in separate messages is accepted by the API
and quietly teaches the model to stop asking in parallel - the results stop
lining up with the request that produced them. One message, every result,
including the ones that failed: a tool that raised comes back as a tool_result
marked is_error, never as a dropped block, because a missing result for a
tool_use id is a malformed conversation.

The iteration cap is a safety rail, not a feature. If it ever fires, the turn
is returned with what it has and the results file records that it was cut off.
"""

from __future__ import annotations

from API.harness import say
from API.harness import tools as harness_tools

MAX_ITERATIONS = 12


def run_turn(call, ctx, history, messages: list[dict]) -> tuple[str, bool]:
    """Drive one turn to completion.

    Input:
        call: a function taking the message list and returning an API response.
        ctx: the shared tool context.
        history: the append-only transcript.
        messages: the conversation so far, mutated as the turn proceeds.

    Output: the model's final text, and whether the turn hit the iteration cap.
    """
    for iteration in range(1, MAX_ITERATIONS + 1):
        response = call(messages)
        assistant = {"role": "assistant", "content": response.content}
        stored = _store(history, assistant)
        messages.append(stored)

        if response.stop_reason != "tool_use":
            say("loop.py", f"stop_reason={response.stop_reason} after {iteration} model calls")
            return _text(response), False

        requested = [block for block in response.content if block.type == "tool_use"]
        say(
            "loop.py",
            f"iteration {iteration}: stop_reason=tool_use, "
            f"{len(requested)} tool(s) requested: {', '.join(b.name for b in requested)}",
        )

        results = [harness_tools.dispatch(ctx, block) for block in requested]
        say("loop.py", f"returning {len(results)} tool results in one user message")
        user_message = {"role": "user", "content": results}
        history.append(user_message)
        messages.append(user_message)

    say("loop.py", f"iteration cap of {MAX_ITERATIONS} reached - turn cut off")
    return "", True


def _store(history, message: dict) -> dict:
    from API.harness.history import to_message

    stored = to_message(message["role"], message["content"])
    history.append(stored)
    return stored


def _text(response) -> str:
    return "\n".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()
