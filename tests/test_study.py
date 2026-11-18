import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from openpyxl import Workbook

from rus.__main__ import main
from rus.data import breast, prepare, reference, row, uclm
from rus.demo import synthetic
from rus.report import report
from rus.routing import above, analyze, calibrate, fit, gate, patient_scores, score, selected
from rus.shared import config, hash_file, read, save
from rus.study import Study
from rus.vision import FEATURES, UNet, crop_box, cut, features, first_pass, letterbox, second_pass


class DataTests(unittest.TestCase):
    def test_clinical_adapters(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images, masks = root/"images", root/"masks"
            images.mkdir()
            masks.mkdir()
            names = ["ALWI_001.png","ALWI_002.png","ALWI_003.png","ALWI_004.png"]
            for name in names:
                Image.new("L",(8,6),20).save(images/name)
            rgb = np.zeros((6,8,3), dtype=np.uint8)
            rgb[2,1,0] = rgb[4,5,1] = 255
            Image.fromarray(rgb).save(masks/names[0])
            metadata = root/"INFO.csv"
            metadata.write_text("Image;Label;Doppler;Marks;Combined\n"+
                                "\n".join([names[0]+";Benign;No;No;No",names[1]+";Normal;No;No;No",
                                          names[2]+";Malignant;No;No;No",names[3]+";Normal;Y;No;No"]))
            accepted, excluded = uclm(images,masks,metadata)
            self.assertEqual((len(accepted),len(excluded)),(2,2))
            self.assertEqual(reference(accepted[0],(6,8)).sum(),2)
            self.assertFalse(reference(accepted[1],(6,8)).any())
            self.assertEqual(accepted[0]["patient"],"uclm:ALWI")
            Image.new("RGB",(9,6)).save(masks/names[2])
            _, geometry_excluded=uclm(images,masks,metadata)
            self.assertIn(["mask_geometry_mismatch"],[item["reasons"] for item in geometry_excluded])
            rgb[1,1,2] = 255
            Image.fromarray(rgb).save(masks/names[0])
            with self.assertRaisesRegex(ValueError,"RGB"):
                reference(accepted[0],(6,8))
            book = Workbook()
            sheet = book.active
            sheet.append(["CaseID","Image_filename","Mask_tumor_filename","Mask_other_filename","Classification"])
            sheet.append([1,names[0],"one.png&two.png","forbidden.png","malignant"])
            sheet.append([2,names[1],None,None,"normal"])
            for num,name in enumerate(("one.png","two.png")):
                target = np.zeros((6,8),dtype=np.uint8)
                target[num,num]=255
                Image.fromarray(target).save(images/name)
            book.save(root/"cases.xlsx")
            records = breast(images,root/"cases.xlsx")
            self.assertEqual(reference(records[0],(6,8)).sum(),2)
            self.assertFalse(reference(records[1],(6,8)).any())
            records[1]["label"]="benign"
            with self.assertRaisesRegex(ValueError,"annotation"):
                reference(records[1],(6,8))

    def test_split_repair_and_quarantine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=synthetic(root/"data")
            rows[0]["fold"]=5
            source=rows[0]
            leaked=row("breast","duplicate",source["image"],source["masks"],source["label"])
            rows=[item for item in rows if item["id"]!=leaked["id"]]+[leaked]
            prepare(rows,root/"run",demo=True)
            manifest=read(root/"run"/"manifest.json")
            self.assertEqual(manifest["split"],"regenerated")
            self.assertIn(leaked["id"],manifest["quarantine"])
            study=Study(root/"run")
            self.assertTrue(study.development)

    def test_crop_and_annotation_free_inference(self):
        torch.set_num_threads(1)
        cfg=config(True)
        mask=np.zeros((31,75),dtype=bool)
        self.assertEqual(crop_box(mask),(0,0,75,31))
        mask[0,0]=mask[-1,-1]=True
        self.assertEqual(crop_box(mask),(0,0,75,31))
        mask[:]=False
        mask[11:19,31:40]=True
        box=crop_box(mask)
        restored=np.zeros_like(mask)
        cut(restored,box)[:]=cut(mask,box)
        np.testing.assert_array_equal(restored,mask)
        tensor,inner=letterbox(mask,32,mask=True)
        self.assertEqual(tuple(tensor.shape),(1,32,32))
        self.assertLess(inner[3]-inner[1],32)
        image=(mask*120).astype(np.uint8)
        model=UNet(4).eval()
        first,box,inputs=first_pass(model,image,cfg,torch.device("cpu"))
        refined=second_pass(model,image,box,cfg,torch.device("cpu"))
        self.assertEqual(first.shape,refined.shape)
        self.assertEqual(set(inputs),set(FEATURES))
        self.assertTrue(np.isfinite(features(np.zeros_like(image,dtype=float),crop_box(first))["entropy"]))


def panel():
    rows=[]
    for index in range(10):
        positive=index!=9
        prob=0.1+index/20
        inputs={name:prob for name in FEATURES}
        rows.append(dict(id=f"id{index}",patient="many" if index<7 else f"p{index}",positive=positive,
                         first=.5,second=.65 if index<7 else .60,features=inputs,gain=.5,difficulty=prob,difficulty_ms=.02,first_ms=1.,
                         second_ms=2.,router_ms=.01,first_miss=False,second_miss=False,first_nonempty=True,
                         second_nonempty=True))
    return rows


class RoutingTests(unittest.TestCase):
    def test_patient_weighted_oracle_includes_normals_in_budget(self):
        rows=panel()
        self.assertEqual(selected(rows,"oracle",.2),{"id7","id8"})
        patients=patient_scores(rows,{"id7","id8"})
        self.assertEqual(patients,{"many":.5,"p7":.6,"p8":.6})
        self.assertIn("id9",selected(rows,"entropy",.2))
        self.assertTrue(gate(rows)["passed"])
        self.assertFalse(gate([dict(r,second=.5) for r in rows])["passed"])

    def test_frozen_threshold_and_ties(self):
        rows=panel()
        picked=selected(rows,"gain",.2)
        self.assertEqual(picked,selected(list(reversed(rows)),"gain",.2))
        self.assertEqual(len(picked),2)
        threshold=calibrate(rows)
        self.assertEqual({r["id"] for r in rows if above(r["gain"],r["id"],threshold)},picked)
        shifted=[dict(r,gain=1.) for r in rows]
        self.assertEqual(len(selected(shifted,"threshold",.2,threshold)),10)
        results=analyze(rows,threshold,100)
        self.assertEqual(results["positive_patients"],3)
        self.assertEqual(results["policies"]["never"]["normal_fp_rate"],1.)

    def test_serialized_router_uses_only_first_pass(self):
        rows=panel()
        model=fit(rows)
        prediction=score(model,rows[0]["features"])
        rows[0].update(id="unseen",patient="changed",second=-10.)
        self.assertEqual(prediction,score(model,rows[0]["features"]))

    def test_batch_selection_without_reference_labels(self):
        rows=panel()
        blinded=[{key:r[key] for key in ("id","gain","difficulty","features")} for r in rows]
        for policy in ("gain","difficulty","entropy","random","always","never"):
            self.assertEqual(selected(rows,policy,.2),selected(blinded,policy,.2))

    def test_normal_frames_do_not_count_as_positive_dice(self):
        rows=panel()
        rows[-1].update(first=1.,second=0.)
        self.assertNotIn(rows[-1]["patient"],patient_scores(rows,set()))

    def test_secondary_target_and_equal_budget(self):
        rows=panel()
        difficulty=fit(rows,"difficulty")
        self.assertAlmostEqual(score(difficulty,rows[0]["features"]),.5)
        self.assertEqual(fit([dict(r,second=0.) for r in rows],"difficulty"),difficulty)
        self.assertNotEqual(fit(rows)["start"],difficulty["start"])
        for index,record in enumerate(rows):
            record["gain"]=float(index in (7,8))
            record["difficulty"]=float(index in (7,9))
        result=analyze(rows,calibrate(rows),100)
        secondary=result["gain_vs_difficulty"]
        self.assertEqual((secondary["common"],secondary["gain_only"],secondary["difficulty_only"]),(1,1,1))
        self.assertAlmostEqual(secondary["delta"],.10/3)
        self.assertEqual(result["policies"]["difficulty"]["retries"],2)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.folder=Path(cls.tmp.name)/"run"
        with contextlib.redirect_stdout(io.StringIO()):
            main(["demo","--run",str(cls.folder)])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_replay_is_deterministic(self):
        target=self.folder/"report"/"metrics.json"
        original=hash_file(target)
        report(self.folder)
        self.assertEqual(original,hash_file(target))
        metrics=read(target)
        self.assertEqual(set(metrics),{"internal","breast","uclm"})
        self.assertEqual(metrics["breast"]["policies"]["gain"]["retries"],2)
        self.assertEqual(metrics["breast"]["policies"]["difficulty"]["retries"],2)
        settings=read(self.folder/"router.json")
        self.assertEqual(settings["model"]["features"],settings["difficulty_model"]["features"])
        self.assertEqual(settings["difficulty_model"]["objective"],"difficulty")
        self.assertEqual(metrics["breast"]["stream_timing"]["retries"],metrics["breast"]["policies"]["threshold"]["retries"])
        self.assertIn("not study evidence",(self.folder/"report"/"REPORT.md").read_text())

    def test_crossfit_isolation_and_freeze(self):
        for fold in (1,2,3):
            prov=read(self.folder/"crossfit"/str(fold)/"provenance.json")
            self.assertFalse(set(prov["training_patients"]) & set(prov["heldout_patients"]))
            self.assertNotIn(fold,prov["folds"])
        study=Study(self.folder)
        with self.assertRaisesRegex(ValueError,"frozen"):
            study.train()
        with self.assertRaisesRegex(ValueError,"already completed"):
            study.evaluate()
        with patch.object(study,"fingerprint",return_value={}):
            with self.assertRaisesRegex(ValueError,"changed"):
                study.evaluate()

    def test_cpu_clinical_guard(self):
        from rus.vision import select_device
        with patch("torch.cuda.is_available",return_value=False):
            with self.assertRaisesRegex(RuntimeError,"CUDA"):
                select_device(False)

    def test_checkpoint_resume(self):
        from rus.vision import train_one
        study=Study(self.folder)
        path=study.pair/"first.pt"
        before=hash_file(path)
        train_one(study.development,"first",study.cfg,torch.device("cpu"),4,path)
        self.assertEqual(before,hash_file(path))
        with self.assertRaisesRegex(ValueError,"differ"):
            train_one(study.calibration,"first",study.cfg,torch.device("cpu"),4,path)

    def test_failed_clinical_pilot_cannot_crossfit(self):
        study=Study(self.folder)
        pred=read(self.folder/"pilot"/"predictions.json")
        for record in pred:
            record["second"]=record["first"]
        with tempfile.TemporaryDirectory() as tmp:
            study.run=Path(tmp)
            study.demo=False
            path=study.run/"pilot"
            save(path/"predictions.json",pred)
            save(path/"gate.json",dict(**gate(pred),predictions=hash_file(path/"predictions.json")))
            save(path/"provenance.json",read(self.folder/"pilot"/"provenance.json"))
            with self.assertRaisesRegex(ValueError,"stopped at pilot"):
                study.crossfit()


if __name__=="__main__":
    unittest.main()
