"""What goes into this turn, and - more to the point - what does not.

The workspace holds three projects, nineteen tasks and seven people, and that is
the toy version; the application this agent was designed against holds hundreds.
Sending the board every turn is the obvious implementation and the wrong one: it
is most of the bill, it pushes the relevant row into the middle of a list, and it
grows without limit while the question stays the same size.

So a turn carries three things and no more. A digest of the workspace, four
lines, counts rather than rows. The memory records that match what the person
just said, chosen by memory.py. And the person's own text. Everything else the
model has to ask for, which it does through search_workspace - and asking is
cheap, because the answers come back as pipe-separated rows.

The whole block goes in the user message rather than the system prompt. The
system prefix is frozen so it can be cached and so the conversation's thinking
blocks stay valid; per-turn context belongs below that line, not above it.
"""

from __future__ import annotations

from API.harness import say


def digest(client) -> str:
    """Four lines on the state of the workspace, recomputed each turn."""
    projects = client.projects()
    open_tasks = client.tasks(limit=200)
    late = [t for t in open_tasks if t.days_late]
    active = [p for p in projects if p.status != "done"]
    lines = [
        f"פרויקטים פעילים: {len(active)} מתוך {len(projects)}",
        f"משימות גלויות: {len(open_tasks)}, מתוכן באיחור: {len(late)}",
    ]
    for project in active[:3]:
        lines.append(f"  {project.id}|{project.name}|פתוחות {project.open_tasks}|"
                     f"באיחור {project.late_tasks}")
    return "\n".join(lines)


def build_turn(client, memory, text: str) -> dict:
    """Assemble the user message for one turn.

    Input:
        client: the workspace, for the digest.
        memory: the harness memory, for the records worth injecting.
        text: what the person said.

    Output: a message dict ready to append to the conversation.
    """
    parts = ["[מצב נוכחי]", digest(client)]
    records = memory.for_turn(text)
    if records:
        parts += ["", "[כללים והחלטות שמורים שרלוונטיים לבקשה הזאת]", records]
    parts += ["", "[הבקשה]", text]
    content = "\n".join(parts)
    say(
        "context.py",
        f"turn context built: {len(content)} chars "
        f"({'with' if records else 'without'} memory records, digest only - no task rows)",
    )
    return {"role": "user", "content": content}
