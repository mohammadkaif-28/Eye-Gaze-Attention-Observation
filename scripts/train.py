"""CPU-default supervised training; resume, validation selection and early stopping."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from gaze.learning import GazeNet, TRANSFORM, angular_error, load_checkpoint


class GazeDataset(Dataset):
    def __init__(self, manifest):
        self.manifest = Path(manifest).resolve()
        with self.manifest.open(encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f"Empty dataset: {manifest}")
        for row in self.rows:
            path = (self.manifest.parent/row["path"]).resolve()
            if not path.is_relative_to(self.manifest.parent) or not path.is_file():
                raise ValueError(f"Missing or unsafe image: {row['path']}")
            vec = np.array([float(row[k]) for k in ("x","y","z")])
            if not np.isfinite(vec).all() or abs(np.linalg.norm(vec)-1) > .01:
                raise ValueError("Manifest contains invalid/non-unit gaze labels")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row = self.rows[index]
        try:
            with Image.open(self.manifest.parent/row["path"]) as image:
                x = TRANSFORM(image.convert("RGB"))
            return x,torch.tensor([float(row[k]) for k in ("x","y","z")],dtype=torch.float32)
        except OSError as exc:
            raise RuntimeError(f"Unreadable training image {row['path']}: {exc}") from exc


def check_disjoint(a,b):
    for columns in (("path",),("recording","identity")):
        left = {tuple(row[k] for k in columns) for row in a.rows}
        right = {tuple(row[k] for k in columns) for row in b.rows}
        if left & right:
            raise ValueError(f"Split leakage in {columns}; use official splits")


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    errors=[]
    for x,y in loader:
        errors.extend(angular_error(model(x.to(device)),y.to(device)).cpu().tolist())
    if not errors or not np.isfinite(errors).all():
        raise ValueError("Evaluation produced no finite metrics")
    return dict(samples=len(errors),mean_angular_error_deg=float(np.mean(errors)),
                median_angular_error_deg=float(np.median(errors)),
                p90_angular_error_deg=float(np.percentile(errors,90)))


def atomic_save(value,path):
    temporary=path.with_suffix(".tmp")
    torch.save(value,temporary)
    temporary.replace(path)


def train(args):
    if min(args.epochs,args.batch_size,args.patience,args.threads) < 1 or args.lr <= 0:
        raise ValueError("epochs, batch size, patience, threads and learning rate must be positive")
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device(args.device)
    root=Path(args.data)
    train_set,val_set=GazeDataset(root/"train.csv"),GazeDataset(root/"val.csv")
    check_disjoint(train_set,val_set)
    fingerprint=hashlib.sha256((root/"train.csv").read_bytes()+(root/"val.csv").read_bytes()).hexdigest()
    generator=torch.Generator().manual_seed(args.seed)
    train_loader=DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=0,generator=generator)
    val_loader=DataLoader(val_set,batch_size=args.batch_size,num_workers=0)
    model=GazeNet().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.0001)
    start,best,stale=0,float("inf"),0
    output=Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Run folder is not empty. Use a new --output or --resume last.pt")
    output.mkdir(parents=True,exist_ok=True)
    if args.resume:
        if Path(args.resume).resolve().parent != output.resolve():
            raise ValueError("Resume into the same --output folder to preserve best.pt and training history")
        model,data=load_checkpoint(args.resume)
        model.to(device)
        if data["dataset_fingerprint"] != fingerprint:
            raise ValueError("Resume dataset differs from checkpoint")
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.0001)
        optimizer.load_state_dict(data["optimizer"])
        start,best,stale=data["epoch"],data["best"],data["stale"]
        torch.set_rng_state(data["rng"])
        generator.set_state(data["loader_rng"])
    if start >= args.epochs:
        raise ValueError("--epochs is the total target and must exceed the resumed epoch")
    (output/"config.json").write_text(json.dumps(vars(args),indent=2),encoding="utf-8")
    for epoch in range(start,args.epochs):
        model.train(); loss_sum=0; n=0; begin=time.monotonic()
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction=model(x)
            loss=(1-(prediction*y).sum(-1)).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss. Last completed epoch remains available.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            loss_sum+=loss.item()*len(x); n+=len(x)
        metrics=evaluate(model,val_loader,device)
        improved=metrics["mean_angular_error_deg"] < best
        best=min(best,metrics["mean_angular_error_deg"])
        stale=0 if improved else stale+1
        checkpoint=dict(architecture="gazenet-v1",coordinates="gaze360",model=model.state_dict(),
            optimizer=optimizer.state_dict(),epoch=epoch+1,best=best,stale=stale,
            dataset_fingerprint=fingerprint,rng=torch.get_rng_state(),loader_rng=generator.get_state(),
            validation=metrics)
        atomic_save(checkpoint,output/"last.pt")
        if improved:
            atomic_save(checkpoint,output/"best.pt")
        entry=dict(epoch=epoch+1,train_cosine_loss=loss_sum/n,seconds=time.monotonic()-begin,**metrics)
        with (output/"history.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(entry)+"\n")
        print(json.dumps(entry),flush=True)
        if stale >= args.patience:
            print("Early stopping: validation did not improve."); break


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",default="data/processed")
    p.add_argument("--output",default="runs/gaze360")
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--patience",type=int,default=5)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--resume")
    try: train(p.parse_args())
    except KeyboardInterrupt: sys.exit("Stopped. Resume from last.pt (last completed epoch).")
    except Exception as exc: sys.exit(f"Training failed: {exc}")
