# tests/test_search.py
from src.vectorstore import search


def main():
    query = input("Enter your question: ")

    results = search(query, k=3)

    for rank, r in enumerate(results, start=1):
        print(f"\nResult {rank} (distance={r['distance']:.4f}, chunk #{r['index']}):\n")
        print(r["text"][:500])  # show first 500 chars of each chunk
        print("\n" + "-" * 80)


if __name__ == "__main__":
    main()
