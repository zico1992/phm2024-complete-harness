from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
from operational.runtime import current_versions,check_versions,check_model_environment
from operational.core import freeze


class RuntimeContracts(unittest.TestCase):
    def test_matching_runtime_is_json_only(self):
        with patch('operational.runtime.joblib.load') as load:
            result=check_versions(current_versions())
        self.assertEqual(result['status'],'passed');self.assertFalse(result['model_loaded']);load.assert_not_called()

    def test_freeze_mismatch_stops_before_unpickling(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);(tmp/'selected_model.joblib').write_bytes(b'not a pickle')
            expected=current_versions();expected['sklearn']='wrong-version'
            (tmp/'environment.json').write_text(json.dumps(expected))
            with patch('operational.runtime.joblib.load') as load:
                with self.assertRaisesRegex(ValueError,'BEFORE model loading'):
                    freeze(tmp/'selected_model.joblib','unused.csv',tmp/'out','test')
                load.assert_not_called()
            self.assertFalse((tmp/'out').exists())

    def test_missing_metadata_is_not_inferred_by_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('operational.runtime.joblib.load') as load:
                with self.assertRaisesRegex(ValueError,'metadata missing'):check_model_environment(Path(tmp)/'model.joblib')
                load.assert_not_called()

    def test_python_minor_mismatch_is_reported(self):
        expected=current_versions();expected['python']='0.1.0'
        with self.assertRaisesRegex(ValueError,'python: expected'):check_versions(expected)

    def test_legacy_missing_auxiliary_versions_are_explicit(self):
        expected={k:v for k,v in current_versions().items() if k in ['python','numpy','pandas','sklearn']}
        result=check_versions(expected)
        self.assertIn('scipy',result['unrecorded_packages'])
