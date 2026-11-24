import pytest
import torch
import yaml

from src.toy_regression import train_v3_members as batched
from src.toy_regression.experiment_config import Config
from src.toy_regression.generate_cubic_data import generate_data, fgsm_epsilon
from src.toy_regression.run_v3_sweep import CONFIG, ROOT, DEFAULT_RUN_ROOT
from src.toy_regression.train_v1_v2_models import CONDITIONS, train_member


def test_matrix_and_output_routing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    matrix = yaml.safe_load(CONFIG.read_text())
    assert DEFAULT_RUN_ROOT == ROOT / "results/raw/toy_regression"
    assert matrix["seeds"] == [42, 123, 2026]
    assert list(matrix["arms"])[-1] == "v3_n_doug_mixed_lr_b1_e500"
    assert len(matrix["arms"]) * 3 * 12 == 504
    c, k = [next(v for key, v in matrix["arms"].items() if key.startswith("v3_"+arm+"_")) for arm in ("c","k")]
    assert c["max_epochs"] * 20 // c["batch_size"] == k["max_epochs"] * 20 // k["batch_size"] == 10000


@pytest.mark.parametrize("condition", CONDITIONS)
def test_member_matches_original_training(tmp_path,condition):
    torch.set_num_threads(1)
    cfg = Config(outer_seeds=(42,),max_epochs=2,batch_size=5,learning_rate=.001)
    splits = {42:generate_data(cfg,42)}
    batched.train_group(tmp_path,"v3_test",condition,cfg,splits,torch.device("cpu"))
    folder = tmp_path/"scalar"
    (folder/"histories").mkdir(parents=True)
    (folder/"models").mkdir()
    split=splits[42]["train"]
    eps=fgsm_epsilon(split.x,cfg.fgsm_epsilon_fraction)
    scalar=train_member(condition,cfg,split,torch.device("cpu"),eps,42,0,folder,"test")
    saved=torch.load(tmp_path/"seed_42"/condition.name/"models/member_0.pt", weights_only=True)
    for k,param in scalar.state_dict().items():
        torch.testing.assert_close(saved["model_state_dict"][k],param,atol=2e-6,rtol=1e-5)
    original=torch.load(folder/"models/member_0.pt",weights_only=True)
    assert torch.equal(original["shuffle_rng_state"],saved["shuffle_rng_state"])
    assert saved["global_step"]==8


def test_resume_mid_training_matches_uninterrupted(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    cfg=Config(outer_seeds=(42,123),max_epochs=101,batch_size=20)
    splits={s:generate_data(cfg,s) for s in cfg.outer_seeds}
    condition=CONDITIONS[2]
    dirs=[tmp_path/name for name in ("once","interrupted")]
    for path in dirs:
        path.mkdir()
    args=("v3_resume",condition,cfg,splits,torch.device("cpu"))
    batched.train_group(dirs[0],*args)
    original=batched.atomic_save
    def interrupt(state,path):
        original(state,path)
        if state.get("epoch")==100:
            raise RuntimeError("simulated interruption")
    with monkeypatch.context() as scoped:
        scoped.setattr(batched,"atomic_save",interrupt)
        with pytest.raises(RuntimeError,match="interruption"):
            batched.train_group(dirs[1],*args)
    batched.train_group(dirs[1],*args)
    for seed in cfg.outer_seeds:
        files=[d/f"seed_{seed}"/condition.name/"models/member_0.pt" for d in dirs]
        a,b=[torch.load(f,weights_only=True) for f in files]
        assert a["global_step"]==b["global_step"]==101
        assert torch.equal(a["shuffle_rng_state"],b["shuffle_rng_state"])
        for k,v in a["model_state_dict"].items():
            assert torch.equal(v,b["model_state_dict"][k])

