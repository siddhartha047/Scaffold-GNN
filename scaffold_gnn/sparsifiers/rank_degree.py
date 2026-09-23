import random
import networkx as nx
import networkit as nk
from torch_geometric.data import Data
from .base import BaseSparsifier


class RankDegreeSparsifier(BaseSparsifier):
    def __init__(self, rho=0.1, seed=None, target_ratio=None):
        self.rho = float(rho)
        self.seed = seed
        self.target_ratio = target_ratio

    def sparsify(self, data_or_graph):
        if isinstance(data_or_graph, Data):
            G_nk, node_list = self._ensure_nk(data_or_graph)
        else:
            G_nk, node_list = self._nx_to_nk(self._ensure_nx(data_or_graph))

        num_edges = G_nk.numberOfEdges()
        num_nodes = G_nk.numberOfNodes()

        if num_edges == 0:
            if isinstance(data_or_graph, Data):
                return self._nk_to_pyg(G_nk, data_or_graph)
            H_nx = nx.DiGraph() if G_nk.isDirected() else nx.Graph()
            H_nx.add_nodes_from(node_list)
            return H_nx

        if self.target_ratio is not None:
            target_ratio = max(1e-6, min(1.0, float(self.target_ratio)))
        else:
            target_ratio = 1.0

        target_num = max(1, min(num_edges, round(num_edges * target_ratio)))

        new_G = nk.graph.Graph(
            num_nodes,
            weighted=G_nk.isWeighted(),
            directed=G_nk.isDirected()
        )

        rng = random.Random(self.seed)
        seeds = rng.sample(range(num_nodes), min(10, num_nodes))
        seen = set()
        rho = self.rho
        iter_count = 0
        count = 0
        max_iters = 10000
        deg = [G_nk.degree(u) for u in range(num_nodes)]

        while new_G.numberOfEdges() < target_num and iter_count < max_iters:
            iter_count += 1

            if not seeds:
                seeds = rng.sample(range(num_nodes), min(10, num_nodes))

            if iter_count % 500 == 0:
                seen.clear()

            if count > int(0.01 * target_num):
                iter_count = 0
                count = 0

            elif iter_count > 1000 and count <= int(0.01 * target_num):
                iter_count = 0
                count = 0
                if rho < 0.99:
                    rho = min(0.99, rho + 0.1)
                    seen.clear()
                else:
                    nodes_by_degree = sorted(
                        range(num_nodes),
                        key=lambda u: deg[u],
                        reverse=True
                    )
                    for u in nodes_by_degree:
                        if new_G.numberOfEdges() >= target_num:
                            break
                        missing = [
                            v for v in G_nk.iterNeighbors(u)
                            if not new_G.hasEdge(u, v)
                        ]
                        missing.sort(key=lambda v: deg[v], reverse=True)
                        for v in missing:
                            if new_G.numberOfEdges() >= target_num:
                                break
                            if G_nk.isWeighted():
                                new_G.addEdge(u, v, w=G_nk.weight(u, v))
                            else:
                                new_G.addEdge(u, v)
                    break

            new_seeds = []
            for node in seeds:
                if new_G.numberOfEdges() >= target_num:
                    break
                if node in seen:
                    continue
                seen.add(node)

                neighbors = list(G_nk.iterNeighbors(node))
                neighbors = sorted(neighbors, key=lambda x: deg[x], reverse=True)

                neighbors = neighbors[:max(1, int(len(neighbors) * rho))]

                for neighbor in neighbors:
                    if new_G.numberOfEdges() >= target_num:
                        break
                    if not new_G.hasEdge(node, neighbor):
                        if G_nk.isWeighted():
                            new_G.addEdge(node, neighbor, w=G_nk.weight(node, neighbor))
                        else:
                            new_G.addEdge(node, neighbor)
                        new_seeds.append(neighbor)
                        count += 1

            seeds = new_seeds

        if isinstance(data_or_graph, Data):
            return self._nk_to_pyg(new_G, data_or_graph)

        H_nx = nx.DiGraph() if G_nk.isDirected() else nx.Graph()
        H_nx.add_nodes_from(node_list)

        for u, v in new_G.iterEdges():
            if new_G.isWeighted():
                H_nx.add_edge(node_list[u], node_list[v], weight=new_G.weight(u, v))
            else:
                H_nx.add_edge(node_list[u], node_list[v])

        return H_nx
