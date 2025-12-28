# tests/test_qa.py

from src.rag import answer_with_llm, get_relevant_context


def main():
    # Simple loop: ask one question, print answer + (optionally) context
    query = input("Enter your question: ").strip()

    if not query:
        print("Empty question, exiting.")
        return

    answer, context = answer_with_llm(query, k=3)

    print("\n=== ANSWER ===")
    print(answer)
    print("\n=== CONTEXT (retrieved chunks) ===")
    print(context)


if __name__ == "__main__":
    main()
