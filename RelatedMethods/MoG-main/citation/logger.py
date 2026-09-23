from ast import arg
import torch

from scripts.common.baseline_result_utils import append_baseline_result


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]
        

    def add_result(self, run, result):
        assert len(result) in [4, 6]
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)       

        #print(self.results) 

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'Highest Test: {result[:, 2].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'  Final Valid: {result[argmax, 1]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
            print(f' Final Kept Ratio: {result[argmax, 3]:.2f}%')
            if isinstance(self.info, dict) and self.info.get('k_list'):
                print(
                    f"[TargetRatio] requested_kept={self.info['k_list'][0]:.8f} "
                    f"achieved_kept={result[argmax, 3].item() / 100.0:.8f}"
                )
            if result.shape[1] >= 6:
                print(f'   Final Test F1 (Macro): {result[argmax, 5]:.2f}')
            append_baseline_result(
                method='mog',
                dataset=self.info.get('dataset') if isinstance(self.info, dict) else None,
                run=run + 1,
                seed=self.info.get('seed') if isinstance(self.info, dict) else None,
                epochs=self.info.get('epochs') if isinstance(self.info, dict) else None,
                kept_ratio=self.info.get('k_list', [''])[0] if isinstance(self.info, dict) and self.info.get('k_list') else None,
                sparsity=100.0 - result[argmax, 3].item(),
                train_acc=result[argmax, 0].item(),
                valid_acc=result[argmax, 1].item(),
                test_acc=result[argmax, 2].item(),
                train_f1_macro=result[argmax, 4].item() if result.shape[1] >= 6 else None,
                test_f1_macro=result[argmax, 5].item() if result.shape[1] >= 6 else None,
                chosen_epoch=argmax,
                # Records whether the *_acc columns hold accuracy or ROC-AUC, so
                # the harvested table cannot silently mix the two.
                metric=self.info.get('metric') if isinstance(self.info, dict) else None,
            )
        else:
            print(self.results)
            
            result = 100 * torch.tensor(self.results)

            print(result)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                best_idx = r[:, 1].argmax()
                train2 = r[best_idx, 0].item()
                test = r[best_idx, 2].item()
                if r.shape[1] >= 6:
                    train_f1 = r[best_idx, 4].item()
                    test_f1 = r[best_idx, 5].item()
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
