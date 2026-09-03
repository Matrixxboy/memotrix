from pathlib import Path

from memotrix import Memory
from memotrix.embeddings import FakeEmbeddings
from memotrix.utils.models import DocumentData


def test_hybrid_flow():
    embeddings = FakeEmbeddings(dim=8)
    memory = Memory(embeddings=embeddings, backend="memory")

    doc1 = DocumentData(
        metadata={"filename": "doc1.txt"},
        sections=[{
            "title": "Introduction",
            "content": "This is a document about machine learning and vector databases.",
        }],
        validation={},
    )
    doc2 = DocumentData(
        metadata={"filename": "doc2.csv"},
        tables=[{
            "headers": ["Name", "Age", "Occupation"],
            "text": "A table containing user information.",
        }],
        validation={},
    )
    doc3 = DocumentData(
        metadata={"filename": "image1.jpg"},
        images=[{
            "description": "A golden retriever playing in a green field.",
            "tags": ["dog", "field", "golden retriever"],
            "path": "images/dog.jpg",
        }],
        validation={},
    )

    memory.add_documents([doc1, doc2, doc3])
    results = memory.search("Find information about dogs.", top_k=2)
    assert isinstance(results, list)
    results2 = memory.search("Show me the user information table", top_k=1)
    assert isinstance(results2, list)
    memory.close()


if __name__ == "__main__":
    test_hybrid_flow()
    print("ok")
