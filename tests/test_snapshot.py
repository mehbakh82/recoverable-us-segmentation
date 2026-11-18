import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from rus import snapshot
from rus.shared import save


class SnapshotTests(unittest.TestCase):
    def test_private_manifest_is_not_archived_and_relocation_is_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            row = dict(patient="private-patient", image="/private-site/image.png", masks=["/private-site/mask.png"],
                       image_sha="image-hash", mask_sha=["mask-hash"], fold=1)
            save(run/"manifest.json",dict(rows=[row]))
            receipt = snapshot.create(run)
            self.assertTrue(receipt["manifest_raw_matches"])
            with zipfile.ZipFile(run/snapshot.NAME) as archive:
                self.assertNotIn("manifest.json",archive.namelist())
                meta = json.loads(archive.read("record.json"))
                self.assertEqual(set(meta["manifest"]),{"raw","portable"})
                self.assertFalse(any(name.startswith(("runs/","data/")) for name in archive.namelist()))
            with self.assertRaises(FileExistsError):
                snapshot.create(run)
            save(run/"manifest.json",dict(rows=[dict(row,image="/new/image.png",masks=["/new/mask.png"])]))
            relocated = snapshot.verify(run)
            self.assertFalse(relocated["manifest_raw_matches"])
            self.assertEqual(receipt["archive_sha256"],relocated["archive_sha256"])
            save(run/"manifest.json",dict(rows=[dict(row,fold=2)]))
            with self.assertRaisesRegex(ValueError,"manifest mismatch"):
                snapshot.verify(run)

    def test_source_changes_and_late_creation_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            save(run/"manifest.json",dict(rows=[]))
            source = run/"protocol.txt"
            source.write_text("original")
            with patch.object(snapshot,"sources",return_value={"PROTOCOL.md":source}):
                snapshot.create(run)
                source.write_text("changed")
                with self.assertRaisesRegex(ValueError,"source mismatch"):
                    snapshot.verify(run)
            (run/"pilot").mkdir()
            with self.assertRaisesRegex(ValueError,"after pilot"):
                snapshot.create(run)
