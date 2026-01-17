import argparse
import asyncio
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.apiyi import ask_llm


def main() -> str:
    parser = argparse.ArgumentParser(description="Query APIYi LLM for a quick response.")
    parser.add_argument(
        "--question",
        default="What do you think about JPMC's financial reports over the last five years?",
        help="Question to send to the LLM.",
    )
    parser.add_argument(
        "--env",
        dest="env_path",
        default=None,
        help="Optional path to a .env file with APIYI_* variables.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override (defaults to APIYI_MODEL).",
    )
    args = parser.parse_args()
    answer = asyncio.run(ask_llm(args.question, env_path=args.env_path, model=args.model))
    print(answer)
    return answer


if __name__ == "__main__":
    main()
