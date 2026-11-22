"""Register Head-aware retrospective target lock without opening future labels."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.drs_peft.data import CACHE, ROOT
from scripts.drs_peft.self_reference_protocol import PUBLIC as SELFREF_PUBLIC
METHODS=("WARMUP_ONLY","LORA_12","HEAD_ONLY_TRAIN","HEAD_LORA","HEAD_BRIDGE_TRAIN","HEAD_BRIDGE_LORA","FULL_FT_TUNED")
def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument("--parent",type=Path,default=ROOT/"results/drs_peft/transfer_scope_v2/TARGET_TOTAL_K_LOCK.json"); p.add_argument("--data-root",type=Path,default=CACHE); p.add_argument("--source-public",type=Path,default=SELFREF_PUBLIC); p.add_argument("--public",type=Path,default=ROOT/"results/drs_peft/head_aware_target_v1"); args=p.parse_args(); args.public.mkdir(parents=True,exist_ok=True)
 parent=read(args.parent); tasks=[]
 for task in parent["target_tasks"]: tasks.append({"task_id":task["task_id"],"domain":task.get("domain","MATR" if task["task_id"]=="MATR_POOLED" else "HUST"),"query_ids":task["query_ids"],"support_pool_ids":task["support_pool_ids"],"support_budget":0,"query_count":len(task["query_ids"]),"target_selection_uses":0})
 manifests={}
 for domain in ("MATR","HUST"):
  path=args.data_root/domain/"COMPLETE.json"
  if not path.is_file(): raise FileNotFoundError(path)
  manifest=read(path); manifests[domain]={"manifest_sha256":sha(path),"status":manifest.get("status"),"physical_cells":manifest.get("physical_cells"),"future_shape":manifest.get("future_shape"),"target_future_labels_read":0}
 peft_summary=args.source_public/"head_bridge_peft_v1"/"PEFT_SUMMARY.csv"; fair_summary=args.source_public/"fair_fullft_v1"/"FAIR_FT_SUMMARY.csv"
 lock={"status":"HEAD_AWARE_RETROSPECTIVE_TARGET_LOCKED_BEFORE_PREDICTION","lock_scope":"K0_zero_shot_target_query_only","parent_target_lock_sha256":sha(args.parent),"historical_target_query_exposure":bool(parent.get("historical_target_query_exposure",True)),"target_selection_uses":0,"target_future_labels_read":0,"outer_labels_read":0,"methods":list(METHODS),"source_protocol":{"domain":"MATR","seed":42,"input_channels":12,"source_public_hashes":{"head_aware_peft_summary":sha(peft_summary),"fair_fullft_summary":sha(fair_summary)},"common_warmup_required":True,"selected_state_replay_required":True},"target_tasks":tasks,"data_cache_manifests":manifests,"prediction_lock_required_before_scoring":True,"prediction_status":"PENDING_SOURCE_STATE_REPLAY","target_query_label_access":"forbidden_until_prediction_lock"}
 (args.public/"TARGET_LOCK.json").write_text(json.dumps(lock,indent=2,ensure_ascii=False)+"\n",encoding="utf-8",newline="")
 receipt={"status":"COMPLETE_HEAD_AWARE_RETROSPECTIVE_TARGET_LOCK","target_future_labels_read":0,"outer_labels_read":0,"target_selection_uses":0,"lock_sha256":sha(args.public/"TARGET_LOCK.json"),"parent_target_lock_sha256":sha(args.parent),"methods":list(METHODS),"files":{"TARGET_LOCK.json":sha(args.public/"TARGET_LOCK.json")}}
 (args.public/"LOCK_RECEIPT.json").write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+"\n",encoding="utf-8",newline=""); print(json.dumps(receipt,ensure_ascii=False))
if __name__=="__main__": main()
