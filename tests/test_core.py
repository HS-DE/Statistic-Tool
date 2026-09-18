import unittest

import numpy as np
import pandas as pd

from statistic_tool.cv import calculate_cv_by_group
from statistic_tool.detection import is_detected
from statistic_tool.identification import group_identification_counts, sample_identification_counts
from statistic_tool.overlap import exclusive_regions, inclusive_intersections, region_counts_for_venn


class DetectionTests(unittest.TestCase):
    def test_nonzero_accepts_negative_values(self):
        df = pd.DataFrame({"S1": [1, 0, np.nan, -2, ""]})
        self.assertEqual(is_detected(df, "nonzero")["S1"].tolist(), [True, False, False, True, False])


class IdentificationTests(unittest.TestCase):
    def setUp(self):
        self.abundance = pd.DataFrame(
            {
                "S1": [1, 0, 2],
                "S2": [0, 3, 2],
                "T1": [0, 3, 0],
                "T2": [4, 0, 0],
            }
        )
        self.map = pd.DataFrame(
            {"sample_id": ["S1", "S2", "T1", "T2"], "group": ["A", "A", "B", "B"]}
        )

    def test_sample_counts(self):
        out = sample_identification_counts(self.abundance, "nonzero", self.map)
        self.assertEqual(out["detected_n"].tolist(), [2, 2, 1, 1])

    def test_group_union_counts(self):
        out = group_identification_counts(self.abundance, self.map, "nonzero", include_total=True)
        self.assertEqual(dict(zip(out["group"], out["detected_n"])), {"A": 3, "B": 2, "Total": 3})


class OverlapTests(unittest.TestCase):
    def test_intersection_names_and_regions(self):
        sets = {"list1": {"A", "B", "C"}, "list2": {"B", "C", "D"}}
        inclusive = inclusive_intersections(sets)
        self.assertEqual(inclusive["intersect(list1, list2)"], {"B", "C"})
        exclusive = exclusive_regions(sets)
        self.assertEqual(exclusive["exclusive(list1)"], {"A"})
        self.assertEqual(region_counts_for_venn(sets), [1, 1, 2])


class CVTests(unittest.TestCase):
    def test_skips_small_group_and_calculates_valid_group(self):
        abundance = pd.DataFrame(
            {
                "A1": [10, 10],
                "A2": [12, 0],
                "A3": [11, 10],
                "B1": [4, 5],
                "B2": [5, 6],
            }
        )
        ids = pd.Series(["P1", "P2"])
        mapping = pd.DataFrame(
            {
                "sample_id": ["A1", "A2", "A3", "B1", "B2"],
                "group": ["A", "A", "A", "B", "B"],
            }
        )
        result = calculate_cv_by_group(
            abundance,
            ids,
            mapping,
            rule="nonzero",
            min_group_size=3,
            min_detect_rate=0.7,
        )
        status = result.group_status.set_index("group")
        self.assertEqual(status.loc["A", "status"], "calculated")
        self.assertEqual(status.loc["B", "status"], "skipped")
        self.assertIn("P1", set(result.long["Protein"]))
        self.assertNotIn("P2", set(result.long["Protein"]))


if __name__ == "__main__":
    unittest.main()
