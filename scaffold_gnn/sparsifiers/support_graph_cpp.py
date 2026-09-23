import os
import subprocess
import tempfile
import time
import torch
from torch_geometric.data import Data
from .base import BaseSparsifier

class SupportGraphCppSparsifier(BaseSparsifier):

    def __init__(self, delta=0.4, sampling_mode='full', sample_size=1000, node_congestion=True, seed=1, verbose=False, preserve_self_loops=False, binary_path=None, target_ratio=None, batch_mode='single', batch_size=1, mode='congestion'):
        self.delta = float(delta)
        self.sampling_mode = sampling_mode
        self.sample_size = int(sample_size)
        self.node_congestion = bool(node_congestion)
        self.seed = int(seed) if seed is not None else 1
        self.verbose = bool(verbose)
        self.preserve_self_loops = bool(preserve_self_loops)
        self.target_ratio = target_ratio
        self.batch_mode = batch_mode
        self.batch_size = int(batch_size)
        self.mode = str(mode)
        if binary_path is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            binary_path = os.path.join(repo_root, 'cpp', 'support_graph_sparsifier')
        self.binary_path = binary_path
        self._build_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'cpp', 'build_support_graph_sparsifier.sh'
        )

    @staticmethod
    def _ekey(u, v):
        return (u, v) if u <= v else (v, u)

    def sparsify(self, data_or_graph):
        if not isinstance(data_or_graph, Data):
            raise ValueError('SupportGraphCppSparsifier expects PyG Data')
        if not os.path.exists(self.binary_path):
            print(f'[support_cpp] Binary not found, building via {self._build_script}...')
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            subprocess.run(['bash', self._build_script], check=True, cwd=repo_root)
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(f'Build failed: binary still missing at {self.binary_path}')
        data = data_or_graph
        n = int(data.num_nodes)
        ei = data.edge_index
        t0 = time.time()
        ei_cpu = ei.detach().to('cpu')
        u = ei_cpu[0].to(torch.long)
        v = ei_cpu[1].to(torch.long)
        self_mask = u.eq(v)
        if self_mask.any():
            self_loops = torch.unique(u[self_mask]).tolist()
        else:
            self_loops = []
        non_self = ~self_mask
        if non_self.any():
            a = torch.minimum(u[non_self], v[non_self])
            b = torch.maximum(u[non_self], v[non_self])
            undirected = torch.stack([a, b], dim=1)
            undirected = torch.unique(undirected, dim=0)
            edges_u = undirected[:, 0].tolist()
            edges_v = undirected[:, 1].tolist()
        else:
            edges_u, edges_v = ([], [])
        delta = self.delta
        if self.target_ratio is not None:
            delta = max(0.001, min(1.0, float(self.target_ratio)))
            print(f'[support_cpp] Auto-tuned delta={delta:.4f} for target_ratio={self.target_ratio}', flush=True)
        m_total = len(edges_u) + len(self_loops)
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, 'in.txt')
            out_path = os.path.join(td, 'out.txt')
            with open(in_path, 'w') as f:
                f.write(f'{n} {m_total}\n')
                for uu, vv in sorted(zip(edges_u, edges_v)):
                    f.write(f'{uu} {vv}\n')
                for i in self_loops:
                    f.write(f'{i} {i}\n')
            cmd = [self.binary_path, '--delta', str(delta), '--sampling_mode', str(self.sampling_mode), '--sample_size', str(self.sample_size), '--node_congestion', '1' if self.node_congestion else '0', '--seed', str(self.seed), '--batch_mode', self.batch_mode, '--batch_size', str(self.batch_size), '--mode', self.mode]
            if self.verbose:
                print(f'[support_cpp] n={n} m={m_total} delta={delta} sampling={self.sampling_mode} K={self.sample_size} node_cong={self.node_congestion} algo={self.mode}', flush=True)
            with open(in_path, 'rb') as fin, open(out_path, 'wb') as fout:
                t_run = time.time()
                subprocess.run(cmd, stdin=fin, stdout=fout, check=True)
                t_run_done = time.time()
            with open(out_path, 'r') as f:
                header = f.readline().strip().split()
                if len(header) != 2:
                    raise RuntimeError('Bad output from C++ sparsifier')
                n_out, m_out = (int(header[0]), int(header[1]))
                if n_out != n:
                    raise RuntimeError(f'Node count mismatch: got {n_out}, expected {n}')
                rows = []
                cols = []
                for _ in range(m_out):
                    line = f.readline()
                    if not line:
                        break
                    u, v = map(int, line.strip().split())
                    rows.extend([u, v])
                    cols.extend([v, u])
                if self.preserve_self_loops:
                    for i in self_loops:
                        rows.append(i)
                        cols.append(i)
        out = data.clone()
        out.edge_index = torch.tensor([rows, cols], dtype=ei.dtype, device=ei.device)
        if self.verbose:
            t1 = time.time()
            print(f'[support_cpp] c++_run={t_run_done - t_run:.3f}s total_wrapper={t1 - t0:.3f}s', flush=True)
        return out
