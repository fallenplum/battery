"""Frozen independent N1/N2 experiments with validation-only selection and resume."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from .data import load_bundle, role_indices, TrainOnlyScaler, synthetic_bundle, file_hash
from .model import ENCODERS, DECODERS, TrajectoryNet

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("n012/data.py", "n012/model.py", "n012/run.py", "n012/audit.py",
           "n012/engineering.py", "tests/test_n012.py",
           "configs/N012_config.json", "research/N012_EXECUTION_CONTRACT_ZH.md")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(pending, path)


def save_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    torch.save(value, temporary)
    os.replace(temporary, path)


def bind_json(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError("Registered identity differs at " + str(path))
    else:
        write_json(path, value)


def configure():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Avoid different Transformer kernels between training and evaluation.
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_random(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


@contextmanager
def execution_lock(path):
    import msvcrt
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def cell_mae(pred, y, mask):
    error = torch.where(mask, (pred-y).abs(), 0.)
    count = mask.sum(1)
    return (error.sum(1)/count.clamp_min(1))[count > 0].mean()


def metrics(pred, y, mask, query):
    pred, y, mask = np.asarray(pred, float), np.asarray(y, float), np.asarray(mask, bool)
    if not np.isfinite(pred).all() or not np.isfinite(y[mask]).all():
        raise FloatingPointError("Nonfinite prediction or supported target")
    def score(support):
        count = support.sum(1)
        valid = count > 0
        error = np.where(support, pred-y, 0.)
        return dict(n_cells=int(valid.sum()), n_obs=int(count.sum()),
                    mae=float(np.mean(np.abs(error).sum(1)[valid]/count[valid])) if valid.any() else None,
                    rmse=float(np.mean(np.sqrt(np.square(error).sum(1)[valid]/count[valid]))) if valid.any() else None)
    result = score(mask)
    for label, lo, hi in (("101_200",101,200),("201_500",201,500),
                          ("501_1000",501,1000),("1001_plus",1001,np.inf)):
        result[label] = score(mask & (query[None]>=lo) & (query[None]<=hi))
    return result


def batches(index, batch_size, seed, epoch):
    order = np.random.default_rng(seed+epoch*1009).permutation(index)
    for start in range(0, len(order), batch_size):
        yield order[start:start+batch_size]


def to_tensors(bundle, x, p, rows, device):
    return {"x":torch.as_tensor(x[rows],device=device),
            "point_mask":torch.as_tensor(bundle.mask[rows],device=device),
            "cycle_ids":torch.as_tensor(bundle.cycle[rows],device=device),
            "policy":torch.as_tensor(p[rows],device=device),
            "policy_mask":torch.as_tensor(bundle.policy_mask[rows],device=device),
            "anchor":torch.as_tensor(bundle.anchor[rows],device=device),
            "query":torch.as_tensor(bundle.query,device=device),
            "y":torch.as_tensor(bundle.targets[rows],device=device),
            "mask":torch.as_tensor(bundle.observed[rows],device=device)}


def forward(model, batch):
    return model(**{key:batch[key] for key in ("x","point_mask","cycle_ids","policy","policy_mask","anchor","query")})


def predict(model, bundle, x, p, rows, device, batch_size):
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = to_tensors(bundle,x,p,rows[start:start+batch_size],device)
            values.append(forward(model,batch).cpu().numpy())
    return np.concatenate(values)


def split_digest(bundle, roles):
    identity = {role:[bundle.ids[i] for i in rows] for role,rows in roles.items()}
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def source_hashes():
    return [{"path":name,"sha256":file_hash(ROOT/name)} for name in SOURCES]


def bundle_fingerprint(bundle):
    digest = hashlib.sha256()
    for key in ("x","mask","cycle","policy","policy_mask","anchor","query","targets","observed","eligible"):
        value = np.ascontiguousarray(getattr(bundle,key))
        for payload in (key.encode(),str(value.dtype).encode(),str(value.shape).encode(),value.tobytes()):
            digest.update(len(payload).to_bytes(8,"big"))
            digest.update(payload)
    return digest.hexdigest()


def unit_path(output, bundle, fold, encoder, decoder):
    return Path(output)/bundle.domain/fold/(encoder+"__"+decoder)


def verify_files(root, receipt):
    for item in receipt["files"]:
        if file_hash(root/item["path"]) != item["sha256"]:
            raise RuntimeError("Sealed artifact changed at "+str(root/item["path"]))


def file_records(root, names):
    return [{"path":name,"sha256":file_hash(root/name)} for name in names]


def train_unit(bundle, fold, encoder, decoder, output, registration, seed=42, epochs=150,
               batch_size=8, lr=3e-4, weight_decay=.01, device="cpu", width=64):
    out = unit_path(output,bundle,fold,encoder,decoder)
    out.mkdir(parents=True,exist_ok=True)
    roles = role_indices(bundle,fold)
    if any(len(rows)==0 for rows in roles.values()):
        raise ValueError("Empty execution role")
    contract = dict(domain=bundle.domain,fold=fold,encoder=encoder,decoder=decoder,seed=seed,epochs=epochs,
                    batch_size=batch_size,learning_rate=lr,weight_decay=weight_decay,width=width,
                    gradient_clip=1.,loss="cell_weighted_masked_future_MAE",split_sha256=split_digest(bundle,roles),
                    roles={role:[bundle.ids[i] for i in rows] for role,rows in roles.items()},
                    study_sha256=registration,expected_updates=epochs*math.ceil(len(roles["train"])/batch_size))
    bind_json(out/"UNIT_CONTRACT.json",contract)
    marker = out/"TRAINING_COMPLETE.json"
    if marker.exists():
        receipt = json.loads(marker.read_text(encoding="utf-8"))
        verify_files(out,receipt)
        return receipt
    scaler = TrainOnlyScaler().fit(bundle,roles["train"])
    x,p = scaler.apply(bundle)
    bind_json(out/"SCALER_TRAIN_ONLY.json",dict(training_ids=contract["roles"]["train"],statistics=scaler.to_dict()))
    seed_all(seed)
    model = TrajectoryNet(encoder,decoder,policy_dim=bundle.policy.shape[1],width=width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    state_path = out/"TRAIN_STATE.pt"
    history, best, checkpoint, updates, start_epoch, elapsed = [], math.inf, None, 0, 0, 0.
    if state_path.exists():
        state = torch.load(state_path,map_location=device,weights_only=False)
        if state["contract"] != contract:
            raise RuntimeError("Resume source, data or settings changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, best, checkpoint = state["history"],state["best_mae"],state["best_checkpoint"]
        updates,start_epoch,elapsed = state["updates"],state["next_epoch"],state["elapsed_seconds"]
        restore_random(state["random_state"])
        save_torch(out/"BEST.pt",checkpoint)
    started = time.monotonic()
    for epoch in range(start_epoch,epochs):
        model.train()
        training_total = 0.
        for rows in batches(roles["train"],batch_size,seed,epoch):
            batch = to_tensors(bundle,x,p,rows,device)
            optimizer.zero_grad(set_to_none=True)
            loss = cell_mae(forward(model,batch),batch["y"],batch["mask"])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite train loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            training_total += float(loss.detach().cpu())*len(rows)
            updates += 1
        val_pred = predict(model,bundle,x,p,roles["validation"],device,batch_size)
        val = metrics(val_pred,bundle.targets[roles["validation"]],bundle.observed[roles["validation"]],bundle.query)["mae"]
        selected = val < best
        if selected:
            best = val
            checkpoint = dict(model={key:value.detach().cpu().clone() for key,value in model.state_dict().items()},
                              contract=contract,best_epoch=epoch+1,validation_mae=val,encoder=encoder,decoder=decoder,
                              policy_dim=bundle.policy.shape[1],width=width,seed=seed)
            save_torch(out/"BEST.pt",checkpoint)
        history.append(dict(epoch=epoch+1,train_mae=training_total/len(roles["train"]),val_mae=val,
                            selected=selected,updates=updates))
        state = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),random_state=random_state(),
                     best_mae=best,best_checkpoint=checkpoint,history=history,updates=updates,next_epoch=epoch+1,
                     contract=contract,elapsed_seconds=elapsed+time.monotonic()-started)
        save_torch(state_path,state)
        write_json(out/"HISTORY.json",history)
        if (epoch+1)%25==0 or epoch==start_epoch:
            print(json.dumps(dict(event="TRAINING",domain=bundle.domain,fold=fold,encoder=encoder,decoder=decoder,
                                  epoch=epoch+1,updates=updates,validation_mae=val)),flush=True)
    receipt = dict(status="TRAINED_VALIDATION_ONLY",unit_contract=contract,validation_mae=best,
                   best_epoch=checkpoint["best_epoch"],updates=updates,
                   parameters=sum(parameter.numel() for parameter in model.parameters()),
                   elapsed_seconds=state["elapsed_seconds"],test_observations_consumed=0,
                   files=file_records(out,["UNIT_CONTRACT.json","SCALER_TRAIN_ONLY.json","HISTORY.json","BEST.pt","TRAIN_STATE.pt"]))
    write_json(marker,receipt)
    del model, optimizer, state
    return receipt


def evaluate_unit(bundle, fold, encoder, decoder, output, selection_path, device="cpu", batch_size=8, **settings):
    out = unit_path(output,bundle,fold,encoder,decoder)
    training = json.loads((out/"TRAINING_COMPLETE.json").read_text(encoding="utf-8"))
    verify_files(out,training)
    selected = json.loads(selection_path.read_text(encoding="utf-8"))
    if fold not in selected["selection_folds"]:
        raise ValueError("Test evaluation lacks a frozen fold selection")
    complete = out/"COMPLETE.json"
    if complete.exists():
        receipt = json.loads(complete.read_text(encoding="utf-8"))
        verify_files(out,receipt)
        return receipt
    contract = training["unit_contract"]
    roles = role_indices(bundle,fold)
    scaler = TrainOnlyScaler().fit(bundle,roles["train"])
    x,p = scaler.apply(bundle)
    model = TrajectoryNet(encoder,decoder,policy_dim=bundle.policy.shape[1],width=contract["width"]).to(device)
    checkpoint = torch.load(out/"BEST.pt",map_location=device,weights_only=False)
    model.load_state_dict(checkpoint["model"])
    prediction = predict(model,bundle,x,p,roles["test"],device,batch_size)
    test = metrics(prediction,bundle.targets[roles["test"]],bundle.observed[roles["test"]],bundle.query)
    cell_rows = [dict(uid=bundle.ids[number],**metrics(prediction[i:i+1],bundle.targets[number:number+1],
                                                     bundle.observed[number:number+1],bundle.query)) for i,number in enumerate(roles["test"])]
    write_json(out/"TEST_CELLS.json",cell_rows)
    temporary = out/"test_predictions.pending.npz"
    np.savez_compressed(temporary,ids=np.asarray(contract["roles"]["test"]),query=bundle.query,
                        prediction=prediction.astype(np.float32),observed=bundle.observed[roles["test"]],target=bundle.targets[roles["test"]])
    os.replace(temporary,out/"test_predictions.npz")
    receipt = dict(training,status="EVALUATED",domain=bundle.domain,fold=fold,encoder=encoder,decoder=decoder,
                   test=test,seed=contract["seed"],epochs=contract["epochs"],selection_frozen_before_test=True,
                   selection_file=selection_path.relative_to(Path(output)).as_posix(),selection_sha256=file_hash(selection_path),
                   training_complete_sha256=file_hash(out/"TRAINING_COMPLETE.json"),
                   files=file_records(out,[item["path"] for item in training["files"]]+["TEST_CELLS.json","test_predictions.npz"]))
    write_json(complete,receipt)
    del model
    return receipt


def pooled_score(records, key="test"):
    total = sum(record[key]["n_cells"] for record in records)
    return dict(n_cells=total,mae=sum(record[key]["mae"]*record[key]["n_cells"] for record in records)/total,
                rmse=sum(record[key]["rmse"]*record[key]["n_cells"] for record in records)/total,
                n_obs=sum(record[key]["n_obs"] for record in records))


def execute(bundle, output, phase="all", fold_limit=None, **settings):
    configure()
    output = Path(output)
    destination = output/bundle.domain
    destination.mkdir(parents=True,exist_ok=True)
    folds = [name for name in bundle.folds if name.startswith("cell_fold_")]
    if fold_limit is not None:
        folds = folds[:fold_limit]
    settings = dict(seed=42,epochs=150,batch_size=8,lr=3e-4,weight_decay=.01,width=64,device="cpu")|settings
    registration = dict(study_id="N012_seed42",domain=bundle.domain,cohort=bundle.ids,
                        tensor_fingerprint=bundle_fingerprint(bundle),sources=source_hashes(),
                        folds={name:bundle.folds[name] for name in folds},settings=settings,
                        encoders=list(ENCODERS),decoders=list(DECODERS),
                        fresh_unique_units=len(folds)*9,query_mlp_N2_reuses_identical_N1_fit=True,
                        selection="strict_first_min_own_fold_validation",historical_exposure=True)
    bind_json(destination/"RUN_REGISTRATION.json",registration)
    registered_hash = file_hash(destination/"RUN_REGISTRATION.json")
    n1_path = destination/"N1_SELECTION.json"
    n2_path = destination/"N2_SELECTION.json"
    if phase in ("N1","all"):
        candidate = {}
        for fold in folds:
            candidate[fold] = {encoder:train_unit(bundle,fold,encoder,"query_mlp",output,registered_hash,**settings)
                               for encoder in ENCODERS}
        chosen = {fold:min(ENCODERS,key=lambda encoder:candidate[fold][encoder]["validation_mae"]) for fold in folds}
        bind_json(n1_path,dict(encoder_per_outer_fold=chosen,selection_folds=folds,
                              criterion="each outer fold inner validation only",nested_selection=True,test_not_used_for_selection=True,
                              study_sha256=registered_hash,
                              candidates={fold:{enc:dict(validation_mae=row["validation_mae"],best_epoch=row["best_epoch"],
                                                        training_complete_sha256=file_hash(unit_path(output,bundle,fold,enc,"query_mlp")/"TRAINING_COMPLETE.json"))
                                                for enc,row in values.items()} for fold,values in candidate.items()}))
    selection = json.loads(n1_path.read_text(encoding="utf-8"))
    if selection["selection_folds"] != folds or selection["study_sha256"] != registered_hash:
        raise RuntimeError("N1 selection identity differs")
    chosen = selection["encoder_per_outer_fold"]
    if phase in ("N2","all"):
        candidate = {}
        for fold in folds:
            candidate[fold] = {decoder:train_unit(bundle,fold,chosen[fold],decoder,output,registered_hash,**settings)
                               for decoder in DECODERS}
        decoders = {fold:min(DECODERS,key=lambda decoder:candidate[fold][decoder]["validation_mae"]) for fold in folds}
        bind_json(n2_path,dict(selected_encoder_per_fold=chosen,selected_decoder_per_fold=decoders,
                              selection_folds=folds,criterion="each outer fold inner validation only",nested_selection=True,
                              test_not_used_for_selection=True,study_sha256=registered_hash,N1_selection_sha256=file_hash(n1_path),
                              candidates={fold:{dec:dict(validation_mae=row["validation_mae"],best_epoch=row["best_epoch"],
                                                        training_complete_sha256=file_hash(unit_path(output,bundle,fold,chosen[fold],dec)/"TRAINING_COMPLETE.json"))
                                                for dec,row in values.items()} for fold,values in candidate.items()}))
    n1_results, n2_results = [], []
    test_gate = n2_path if phase in ("all","N2") else n1_path
    if phase in ("N1","all"):
        for fold in folds:
            for encoder in ENCODERS:
                n1_results.append(evaluate_unit(bundle,fold,encoder,"query_mlp",output,test_gate,**settings))
    if phase in ("N2","all"):
        for fold in folds:
            for decoder in DECODERS:
                n2_results.append(evaluate_unit(bundle,fold,chosen[fold],decoder,output,test_gate,**settings))
    if n1_results:
        write_json(destination/"N1_RESULTS.json",dict(models={enc:pooled_score([row for row in n1_results if row["encoder"]==enc]) for enc in ENCODERS},
                                                     nested=pooled_score([row for row in n1_results if row["encoder"]==chosen[row["fold"]]])))
    if n2_results:
        decoders = json.loads(n2_path.read_text(encoding="utf-8"))["selected_decoder_per_fold"]
        write_json(destination/"N2_RESULTS.json",dict(models={dec:pooled_score([row for row in n2_results if row["decoder"]==dec]) for dec in DECODERS},
                                                     nested=pooled_score([row for row in n2_results if row["decoder"]==decoders[row["fold"]]])))
    paths = sorted(destination.glob("cell_fold_*/*/COMPLETE.json"))
    expected = len(folds)*(9 if phase in ("all","N2") else 5)
    if len(paths) != expected:
        raise RuntimeError("Completed matrix coverage differs")
    write_json(destination/"MATRIX_COMPLETE.json",dict(status="COMPLETE",phase=phase,units=len(paths),
               successful_updates=sum(json.loads(path.read_text())["updates"] for path in paths),
               registration_sha256=registered_hash,files=file_records(destination,[path.relative_to(destination).as_posix() for path in paths])))
    print(json.dumps(dict(event="MATRIX_COMPLETE",domain=bundle.domain,units=len(paths))),flush=True)
    return registration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root",type=Path,default=ROOT)
    parser.add_argument("--domain",choices=("MATR","HUST","Stanford","SYNTHETIC"),default="MATR")
    parser.add_argument("--phase",choices=("N0","N1","N2","all"),default="all")
    parser.add_argument("--out",type=Path,default=Path("results/N012_independent"))
    parser.add_argument("--epochs",type=int,default=150)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--learning-rate",type=float,default=3e-4)
    parser.add_argument("--weight-decay",type=float,default=.01)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--width",type=int,default=64)
    parser.add_argument("--fold-limit",type=int)
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    args = parser.parse_args()
    configure()
    device = "cuda" if args.device=="auto" and torch.cuda.is_available() else "cpu" if args.device=="auto" else args.device
    if args.epochs<1 or args.seed!=42:
        parser.error("This registered execution requires positive epochs and seed42")
    if device=="cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    from .audit import audit_domain
    if args.phase=="N0":
        audit_domain(args.repo_root,args.domain,args.out)
        return
    bundle = synthetic_bundle() if args.domain=="SYNTHETIC" else load_bundle(args.repo_root,args.domain)
    if args.domain!="SYNTHETIC":
        audit_domain(args.repo_root,args.domain,args.out,bundle=bundle)
    with execution_lock(args.out/args.domain/".run.lock"):
        execute(bundle,args.out,phase=args.phase,fold_limit=args.fold_limit,seed=args.seed,epochs=args.epochs,
                batch_size=args.batch_size,lr=args.learning_rate,weight_decay=args.weight_decay,width=args.width,device=device)


if __name__=="__main__":
    main()
