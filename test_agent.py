import os
from agent import graph

def test_graph_compiles():
    assert graph is not None
    assert "__start__" in graph.nodes
    assert "planner_node" in graph.nodes
    assert "parallel_execution_node" in graph.nodes
    assert "synthesizer_node" in graph.nodes

def main():
    print("Testing the Deep Research Agent Graph...")
    print("Nodes:", graph.nodes)
    print("Graph compiled successfully.")

if __name__ == "__main__":
    main()
