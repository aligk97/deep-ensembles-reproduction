from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.toy_regression.compare_regional_performance import collect_region
from src.toy_regression.evaluate_v4_lr_comparison import pair_scores
from src.toy_regression.load_saved_predictions import Samples


def test_paired_seed_difference_and_sample_std():
    base = pd.DataFrame(dict(seed=[42,123,2026], model="nll_single", region="in_domain", metric="rmse_true",
                             distribution="point_or_raw_moments", n_samples=2000, value=[1.,4.,8.]))
    new = base.assign(value=[2.,3.,10.]).iloc[::-1]
    pairs, result = pair_scores(base,new)
    np.testing.assert_array_equal(pairs.delta, [1.,-1.,2.])
    assert result.paired_std_delta.item() == pytest.approx(np.std([1.,-1.,2.],ddof=1))
    assert result.v4_mean.item() == 5
    with pytest.raises(ValueError,match="Unpaired"):
        pair_scores(base,new.iloc[1:])


def test_overall_pool_in_domain_and_both_ood_sides_once():
    x = torch.tensor([-6.,-4.,0.,4.,6.])[:,None]
    y = x**3
    splits=[]
    for indices in ([1,2,3],[0,4]):
        splits.append((Samples(x[indices],y[indices],Path("sample")), {"mean": y[indices],
                       "member_means": y[indices][None,...]}))
    joined, truth, prediction = collect_region(splits,"overall")
    assert joined.flatten().sort().values.tolist()==[-6.,-4.,0.,4.,6.]
    assert torch.equal(truth,prediction["member_means"][0])


def test_primary_matrix_is_single_arm_and_exploratory_requires_separate_config():
    configs = Path(__file__).resolve().parents[1]/"configs"
    for tag,lr in (("0p3",.3),("0p03",.03)):
        matrix = yaml.safe_load((configs/f"v4_lr_{tag}.yaml").read_text(encoding="utf-8"))
        assert list(matrix["arms"])==[f"v4_lr_{tag}"]
        assert matrix["learning_rate"]==lr and matrix["baseline_arm"]=="v2_a"
        assert matrix["seeds"]==[42,123,2026]

