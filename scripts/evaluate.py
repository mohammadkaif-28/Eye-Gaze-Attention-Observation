import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from scripts.train import GazeDataset, check_disjoint, evaluate
from gaze.learning import load_checkpoint

def main():
    p=argparse.ArgumentParser(description="Evaluate selected checkpoint on untouched official test subset")
    p.add_argument("--data",default="data/processed")
    p.add_argument("--checkpoint",default="runs/gaze360/best.pt")
    p.add_argument("--output",default="runs/gaze360/test_metrics.json")
    args=p.parse_args()
    torch.set_num_threads(4)
    root=Path(args.data)
    test=GazeDataset(root/"test.csv")
    for split in ("train","val"):
        check_disjoint(test,GazeDataset(root/f"{split}.csv"))
    model,_=load_checkpoint(args.checkpoint)
    metrics=evaluate(model,DataLoader(test,batch_size=32,num_workers=0),"cpu")
    metrics["scope"]="Frontal, visible-eye Gaze360 subset; not a clinical or full-benchmark validation"
    dest=Path(args.output); dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2))

if __name__ == "__main__":
    try: main()
    except Exception as exc: sys.exit(f"Evaluation failed: {exc}")
