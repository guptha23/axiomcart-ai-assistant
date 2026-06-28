"""
main.py — Stage 4: Interactive REPL + Voice loop (complete AxiomCart).

╔══════════════════════════════════════════════════════════════════╗
║  STAGE 4 CONCEPT: Session Management + Voice I/O                ║
║                                                                  ║
║  AxiomCartAssistant wraps the graph with:                        ║
║                                                                  ║
║  1. thread_id — a UUID per session.  The MemorySaver uses this  ║
║     as the conversation key.  Same thread_id = same session.     ║
║                                                                  ║
║  2. HITL handling — after each invoke(), check for interrupts    ║
║     in result["__interrupt__"].  If present, collect user input  ║
║     and resume with Command(resume=answer).                      ║
║                                                                  ║
║  3. Voice mode — same query() method, different I/O:             ║
║     text in  → microphone (sounddevice) → Whisper STT → text    ║
║     text out → OpenAI TTS → audio bytes → speakers              ║
║     The graph doesn't know or care how input/output happened.    ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python3 -m stage4.main              # text REPL
    python3 -m stage4.main --voice      # voice loop (needs microphone)
    python3 -m stage4.main --query "..."  # single query
"""

from __future__ import annotations

import argparse
import uuid

from langchain.messages import HumanMessage
from langgraph.types import Command

from modules.stage1.config import get_logger
from modules.stage4.graph import axiomcart_graph

logger = get_logger("main")


class AxiomCartAssistant:
    """Wraps the multi-agent graph with HITL handling + optional voice I/O.

    One instance = one conversation session.
    The thread_id ties all turns together via the MemorySaver checkpointer.
    """

    def __init__(self, enable_voice: bool = False, voice: str = "nova"):
        self.enable_voice = enable_voice
        # Every session gets a unique thread_id.
        # The MemorySaver uses this to persist conversation state.
        self.thread_id = uuid.uuid4().hex

        if enable_voice:
            # Import here so text-only runs don't need sounddevice/soundfile
            from src.voice import VoiceRecorder, VoiceSpeaker
            self.recorder = VoiceRecorder()
            self.speaker  = VoiceSpeaker(voice=voice, speed=1.1)
        else:
            self.recorder = None
            self.speaker  = None

        logger.info("AxiomCartAssistant ready  (thread=%s, voice=%s)", self.thread_id[:8], enable_voice)

    def query(self, text: str, input_fn=None) -> str:
        """Send a text query through the multi-agent graph.

        Handles HITL interrupts automatically:
          1. First invoke() may pause at interrupt() inside support_model.
          2. We surface the question to the user (text or voice).
          3. Second invoke(Command(resume=answer)) resumes the graph.

        Args:
            text:     the user's message
            input_fn: override for collecting user input (useful for testing)
        """
        if input_fn is None:
            input_fn = lambda prompt: input(f"\n🔄 Agent asks: {prompt}\nYou: ").strip()

        config = {"configurable": {"thread_id": self.thread_id}}

        # First invocation
        result = axiomcart_graph.invoke(
            {"messages": [HumanMessage(content=text)], "user_query": text},
            config,
        )

        # HITL loop — the graph may pause multiple times (rare but possible)
        while "__interrupt__" in result and result["__interrupt__"]:
            question = result["__interrupt__"][0].value
            logger.info("HITL interrupt: %r", question)

            if self.enable_voice and self.speaker and self.recorder:
                self.speaker.speak(question)
                _, user_answer = self.recorder.record_and_transcribe(duration=5)
                if not user_answer:
                    user_answer = "I don't have that information right now"
            else:
                user_answer = input_fn(question)

            logger.info("HITL resume: user_answer=%r", user_answer)
            result = axiomcart_graph.invoke(Command(resume=user_answer), config)

        answer = result.get("final_answer", "")
        if not answer:
            answer = "Sorry, I wasn't able to process that. Could you try rephrasing?"
        return answer

    def text_loop(self) -> None:
        """Interactive text REPL.

        The checkpointer persists state across turns, so follow-up
        questions like "What about Sony?" work naturally.
        """
        print("\n🛒  AxiomCart Assistant  (type 'quit' to exit)\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
                print("Goodbye!")
                break
            answer = self.query(user_input)
            print(f"\nAssistant: {answer}\n")

    def voice_loop(self, max_turns: int = 10) -> None:
        """Microphone-based conversation loop.

        Same pipeline as text_loop — only I/O differs.
        """
        if not self.recorder or not self.speaker:
            logger.error("Voice components not initialised — run with enable_voice=True")
            return

        welcome = "Hello! I'm your AxiomCart assistant. How can I help you today?"
        print(f"\nAssistant: {welcome}")
        self.speaker.speak(welcome)

        for turn in range(1, max_turns + 1):
            logger.info("--- voice turn %d / %d ---", turn, max_turns)

            _, transcript = self.recorder.record_and_transcribe(duration=5)
            if not transcript:
                self.speaker.speak("I didn't catch that. Could you repeat?")
                continue

            if transcript.lower().strip() in ("goodbye", "bye", "quit", "exit"):
                self.speaker.speak("Goodbye! Have a great day.")
                break

            print(f"\nYou: {transcript}")
            answer = self.query(transcript)
            print(f"Assistant: {answer}\n")
            self.speaker.speak(answer)


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AxiomCart Multi-Agent AI Assistant")
    parser.add_argument("--voice",  action="store_true", help="Use microphone + TTS")
    parser.add_argument("--query",  type=str,            help="Single query and exit")
    args = parser.parse_args()

    assistant = AxiomCartAssistant(enable_voice=args.voice)

    if args.query:
        print(assistant.query(args.query))
    elif args.voice:
        assistant.voice_loop()
    else:
        assistant.text_loop()


if __name__ == "__main__":
    main()
