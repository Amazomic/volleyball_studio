#!/usr/bin/env python3
"""Synthetic validation of the independent reviewed-data exporter."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

src=Path(__file__).resolve().parents[1]/"training"/"export_reviewed_tcn.py"
spec=importlib.util.spec_from_file_location("export_reviewed_tcn", src)
ex=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

def dump(p, value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False),encoding="utf-8")

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)
        self.data=self.base/"data"
        self.project=self.data/"projects"/"project"
        self.replay=self.project/"native_cpu_replays"/"replay"
        self.touch=self.project/"native_cpu_touch_runs"/"touch"
        self.replay.mkdir(parents=True)
        self.touch.mkdir(parents=True)
        frames=[]
        for i in range(25):
            ball={"source_model":"vballnet","detection":{
                "score":0.2 if i==5 else 0.95,
                "bbox_xyxy_norm":[0.2+i/1000,0.3,0.22+i/1000,0.32]
            }}
            if i==7:ball["source_model"]="ravel"
            frames.append({"frame_index":i,"track_ids_provisional":True,
                           "confirmed_athlete_identities":False,
                           "selected_ball":ball})
        raw="".join(json.dumps(v,separators=(",",":"))+"\n" for v in frames).encode()
        (self.replay/"tracked_frames.jsonl").write_bytes(raw)
        sha=hashlib.sha256(raw).hexdigest()
        original={"sha256":"a"*64,"schema":"volley-original-source/1"}
        dump(self.replay/"manifest.json",{
            "schema":"volley-native-cpu-tracking/1","project_id":"project",
            "video":"match.mp4","sha256":sha,"file":"tracked_frames.jsonl",
            "frames":25,"fps":30,"original_source":original
        })
        touch_raw=b'{"frame_index":4}\n'
        (self.touch/"touch_candidates.jsonl").write_bytes(touch_raw)
        t_sha=hashlib.sha256(touch_raw).hexdigest()
        dump(self.touch/"manifest.json",{
            "project_id":"project","video":"match.mp4","sha256":t_sha,
            "source_replay_id":"replay","source_replay_sha256":sha,
            "frames":25,"fps":30,"width":1280,"height":720,
            "original_source":original,"settings":{"min_ball_score":0.35}
        })
        self.label={"schema":ex.ANNOTATION_SCHEMA,"videos":[{
            "project_id":"project","video":"match.mp4","reviewed":True,
            "total_frames":25,"baseline_touch_sha":t_sha,
            "source_sha256":ex.provenance_hash(original),
            "touches":[{"frame":4},{"frame":16},{"frame":24}],
            "rallies":[{"start":0,"end":19},{"start":15,"end":24}]
        }]}
        self.write()
    def write(self):
        dump(self.data/"training"/"touch_annotations.json",self.label)
    def run_export(self):
        target=self.base/"out"/"samples.jsonl"
        report=self.base/"out"/"report.json"
        metadata=ex.export_dataset(self.data,target,report)
        return metadata,[json.loads(row) for row in target.read_text(encoding="utf-8").splitlines()]
    def test_merged_segments_and_non_duplicate_touch(self):
        meta,rows=self.run_export()
        self.assertEqual(meta["contacts"],3)
        self.assertEqual(meta["segments"],1)
        self.assertEqual(meta["videos"][0]["overlap_pairs_merged"],1)
        self.assertFalse(meta["can_holdout_by_source"])
        self.assertEqual(rows[0]["contacts"],[4,16,24])
        self.assertEqual(rows[0]["ball_xy"][5],None)
        self.assertEqual(rows[0]["ball_xy"][7],None)
        self.assertEqual(len(rows[0]["ball_xy"]),25)
    def test_unreviewed_is_rejected(self):
        self.label["videos"][0]["reviewed"]=False
        self.write()
        with self.assertRaisesRegex(ValueError,"no human-reviewed"):
            self.run_export()
    def test_source_hash_mismatch_rejected(self):
        self.label["videos"][0]["source_sha256"]="0"*64
        self.write()
        with self.assertRaisesRegex(ValueError,"provenance"):
            self.run_export()
    def test_mutated_replay_rejected(self):
        with (self.replay/"tracked_frames.jsonl").open("ab") as out:
            out.write(b"\n")
        with self.assertRaisesRegex(ValueError,"SHA"):
            self.run_export()
    def test_label_outside_rally_rejected(self):
        self.label["videos"][0]["rallies"]=[{"start":0,"end":19}]
        self.write()
        with self.assertRaisesRegex(ValueError,"outside"):
            self.run_export()
    def test_refuses_overwriting_output(self):
        self.run_export()
        with self.assertRaises(FileExistsError):
            self.run_export()

if __name__=="__main__":
    unittest.main()
