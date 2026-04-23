from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.analytics import Spark
from diagrams.generic.database import SQL
from diagrams.generic.storage import Storage
from diagrams.programming.language import Python

with Diagram(
    "Data Ingestion Pipeline", 
    filename="data_ingestion_test", 
    show=False,
    direction="TB",
    graph_attr={
        "fontsize": "24",
        "fontname": "Helvetica",
        "pad": "0.5",
        "splines": "spline",
    }
):
    with Cluster("Source Datasets"):
        chatdoc = SQL("ChatDoctor\nHealthCareMagic")
        pubmed = SQL("PubMedQA")
        medmcqa = SQL("MedMCQA")
        sources = [chatdoc, pubmed, medmcqa]

    with Cluster("Pre-processing"):
        fmt = Python("Format\nNormalisation")
        clean = Python("Text\nCleaning")
        chunk = Python("Recursive\nSentenceChunker")
        
    with Cluster("Embedding"):
        embed = Spark("all-MiniLM-L6-v2\n(384-dim)")

    with Cluster("Index Construction"):
        bm25b = Spark("BM25\nBuilder")

    with Cluster("Knowledge Base"):
        chroma = PostgreSQL("ChromaDB")
        bm25c = Storage("BM25 Cache")

    sources >> fmt >> clean >> chunk
    chunk >> Edge(label="plain text") >> embed >> Edge(label="upsert docs+vectors") >> chroma
    chunk >> Edge(label="raw tokens") >> bm25b >> Edge(label="serialize") >> bm25c
