import argparse
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from scipy.io import savemat
import torch
from gaze.learning import vector_direction, angular_error, load_checkpoint
from scripts.preprocess_gaze360 import prepare
from scripts.train import train, GazeDataset, check_disjoint


class LearningTests(unittest.TestCase):
    def test_coordinate_convention(self):
        for vector,direction in [([0,0,-1],"towards camera"),([-.5,0,-1],"right"),
            ([.5,0,-1],"left"),([0,.5,-1],"up"),([0,-.5,-1],"down"),([0,0,1],"indeterminate")]:
            self.assertEqual(vector_direction(vector),direction)
        self.assertAlmostEqual(angular_error(torch.tensor([[1.,0,0]]),torch.tensor([[0.,1,0]])).item(),90)

    def test_synthetic_preprocess_train_resume(self):
        # Artificial images validate plumbing only; no Gaze360 training or accuracy claim.
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/"raw"; root.mkdir()
            n=9
            for i in range(n):
                path=root/"imgs"/f"rec{i//3}"/"head"/"000000"/f"{i:06}.jpg"
                path.parent.mkdir(parents=True,exist_ok=True)
                pixels=np.random.default_rng(i).integers(0,255,(80,80,3),dtype=np.uint8)
                Image.fromarray(pixels).save(path)
            meta=dict(recordings=np.array(["rec0","rec1","rec2"],dtype=object),recording=np.arange(n)//3,
                frame=np.arange(n),person_identity=np.zeros(n),split=np.arange(n)//3,
                gaze_dir=np.tile([0,0,-1.],(n,1)),person_head_bbox=np.tile([0,0,1,1.],(n,1)),
                person_face_bbox=np.tile([.1,.1,.8,.8],(n,1)),
                person_eye_left_bbox=np.tile([.2,.2,.1,.1],(n,1)),
                person_eye_right_bbox=np.tile([.6,.2,.1,.1],(n,1)))
            savemat(root/"metadata.mat",meta)
            processed=Path(folder)/"processed"
            report=prepare(root,processed)
            self.assertEqual(report["counts"],dict(train=3,val=3,test=3))
            with self.assertRaises(ValueError): check_disjoint(GazeDataset(processed/"train.csv"),GazeDataset(processed/"train.csv"))
            output=Path(folder)/"run"
            args=argparse.Namespace(epochs=1,batch_size=2,patience=2,threads=1,lr=.001,seed=42,
                device="cpu",data=str(processed),output=str(output),resume=None)
            train(args)
            model,data=load_checkpoint(output/"best.pt")
            self.assertEqual(data["epoch"],1)
            self.assertEqual(tuple(model(torch.zeros(1,3,128,128)).shape),(1,3))
            args.epochs=2; args.resume=str(output/"last.pt")
            train(args)
            _,data=load_checkpoint(output/"last.pt")
            self.assertEqual(data["epoch"],2)

    def test_missing_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError): prepare(Path(folder)/"missing",Path(folder)/"out")

if __name__ == "__main__": unittest.main()
