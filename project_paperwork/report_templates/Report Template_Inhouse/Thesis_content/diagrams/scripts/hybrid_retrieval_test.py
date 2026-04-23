from diagrams import Diagram, Cluster, Edge
from diagrams.aws.general import User
from diagrams.programming.language import Python
from diagrams.onprem.database import PostgreSQL
from diagrams.generic.storage import Storage
from diagrams.onprem.analytics import Spark
from diagrams.generic.compute import Rack

output_path = "/home/kbs/Documents/final_project/Report Template_Inhouse/Thesis_content/diagrams/scripts/hybrid_retrieval_test"

with Diagram(
    "Hybrid Retrieval Pipeline", 
    filename=output_path, 
    show=False,
    direction="TB",
    graph_attr={
        "fontsize": "24",
        "fontname": "Helvetica",
        "pad": "0.5",
        "splines": "spline",
    }
):
    query = User("User Query")

    with Cluster("Encoding Layer"):
        encoder = Python("Query Encoder\n(all-MiniLM-L6)")
        tokeniser = Python("BM25 Tokeniser")

    with Cluster("Storage Layer"):
        chroma = PostgreSQL("ChromaDB\n(Dense)")
        bm25_idx = Storage("BM25 Index\n(Sparse)")

    with Cluster("Retrieval Results"):
        dense_res = Rack("Dense Results\ntop-k=10")
        sparse_res = Rack("Sparse Results\ntop-k=10")

    with Cluster("Fusion & Filtering"):
        rrf = Spark("Reciprocal Rank\nFusion (RRF)")
        diversity = Python("Source Diversity\nFilter")
        dyn_k = Python("Dynamic k\nSelection")

    ctx = User("Top Documents\n(to Generator)")

    query >> encoder >> chroma >> dense_res >> rrf
    query >> tokeniser >> bm25_idx >> sparse_res >> rrf
    rrf >> diversity >> dyn_k >> ctx

print(f"Written: {output_path}.png")
