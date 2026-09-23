import torch

from scripts.common.baseline_result_utils import append_baseline_result


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) in (3, 5)
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
            append_baseline_result(
                method='dspar',
                dataset=getattr(self.info, 'dataset', None),
                run=run + 1,
                seed=getattr(self.info, 'seed', None),
                epochs=getattr(self.info, 'epochs', None),
                kept_ratio=getattr(self.info, 'kept_ratio', None),
                sparsity=getattr(self.info, 'actual_sparsity', None),
                train_acc=result[argmax, 0].item(),
                valid_acc=result[argmax, 1].item(),
                test_acc=result[argmax, 2].item(),
                train_f1_macro=result[argmax, 3].item() if result.shape[1] >= 5 else None,
                test_f1_macro=result[argmax, 4].item() if result.shape[1] >= 5 else None,
                chosen_epoch=argmax,
                # Records whether the *_acc columns hold accuracy or ROC-AUC, so
                # the harvested table cannot silently mix the two.
                metric=getattr(self.info, 'metric', None),
            )
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                if r.shape[1] >= 5:
                    train_f1 = r[r[:, 1].argmax(), 3].item()
                    test_f1 = r[r[:, 1].argmax(), 4].item()
                    best_results.append((train1, valid, train2, test, train_f1, test_f1))
                else:
                    best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            def _std(x):
                return x.std(unbiased=False).item() if x.numel() > 1 else 0.0
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {_std(r):.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {_std(r):.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {_std(r):.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {_std(r):.2f}')
            if best_result.shape[1] >= 6:
                r = best_result[:, 5]
                print(f'   Final Test F1 (Macro): {r.mean():.2f} ± {_std(r):.2f}')
