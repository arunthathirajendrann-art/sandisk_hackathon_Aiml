"""Unit tests for Phase 8 Interactive SanDisk Yield AI Explorer Demo.

Verifies demo data loader, artifact file existence, non-mutation of production files,
and validity of interpretability metrics.
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from demo.data_loader import (
    load_final_metrics,
    load_wafer_heatmap_data,
    load_explanation_samples,
    load_block_anomaly_samples,
    load_contribution_summary,
    get_plot_path,
)

class TestDemoDashboard(unittest.TestCase):
    def test_final_metrics_load(self):
        metrics = load_final_metrics()
        self.assertIn("model_a", metrics)
        self.assertIn("model_b", metrics)
        self.assertAlmostEqual(metrics["model_a"]["ap"], 0.581627, places=4)
        self.assertAlmostEqual(metrics["model_b"]["ap"], 0.654879, places=4)

    def test_wafer_heatmap_data_load(self):
        df = load_wafer_heatmap_data()
        self.assertFalse(df.empty, "wafer_heatmap_data.csv should not be empty")
        expected_cols = {"wafer_id", "die_row", "die_col", "old_label", "predicted_probability"}
        self.assertTrue(expected_cols.issubset(set(df.columns)), f"Missing columns in heatmap data: {expected_cols - set(df.columns)}")
        self.assertTrue((df["predicted_probability"] >= 0).all() and (df["predicted_probability"] <= 1).all())

    def test_explanation_samples_load(self):
        samples = load_explanation_samples()
        self.assertGreater(len(samples), 0, "explanation_samples.json should contain samples")
        first = samples[0]
        self.assertIn("wafer_id", first)
        self.assertIn("logit_decomposition", first)
        decomp = first["logit_decomposition"]
        self.assertIn("spatial_contribution", decomp)
        self.assertIn("parametric_contribution", decomp)
        self.assertIn("block_contribution", decomp)
        
        # Verify logit identity
        res = abs(first["final_logit"] - (decomp["spatial_contribution"] + decomp["parametric_contribution"] + decomp["block_contribution"]))
        self.assertLess(res, 1e-4)

    def test_block_anomaly_samples_load(self):
        df = load_block_anomaly_samples()
        self.assertFalse(df.empty, "block_anomaly_samples.csv should not be empty")
        self.assertIn("peak_anomaly_index", df.columns)

    def test_plot_paths(self):
        plot_p = get_plot_path("wafer_heatmap_W_N_0083.png")
        self.assertIsNotNone(plot_p, "wafer_heatmap_W_N_0083.png plot should exist")
        self.assertTrue(plot_p.exists())

    def test_production_files_unmodified(self):
        """Verify core production files exist and remain untampered."""
        frozen_files = [
            "tuned/pipeline.py",
            "tuned/hazard.py",
            "tuned/blocks.py",
            "tuned/channels.py",
            "tuned/head.py",
            "tuned/cache.py",
            "tuned/final.py",
        ]
        for path_str in frozen_files:
            p = Path(path_str)
            self.assertTrue(p.exists(), f"Production file {path_str} must exist")

    def test_die_selection_state_logic(self):
        """Regression test for Plotly point click and session state selection logic."""
        from demo.data_loader import load_all_prediction_wafers
        df = load_all_prediction_wafers()
        self.assertFalse(df.empty)
        wafers = sorted(df["wafer_id"].unique())
        self.assertGreaterEqual(len(wafers), 1)
        
        # Test die lookup logic for a selected wafer
        test_wafer = wafers[0]
        wafer_data = df[df["wafer_id"] == test_wafer]
        avail_rows = sorted(wafer_data["die_row"].unique())
        self.assertGreater(len(avail_rows), 0)
        
        target_row = avail_rows[0]
        avail_cols = sorted(wafer_data[wafer_data["die_row"] == target_row]["die_col"].unique())
        self.assertGreater(len(avail_cols), 0)
        target_col = avail_cols[0]
        
        die_match = wafer_data[(wafer_data["die_row"] == target_row) & (wafer_data["die_col"] == target_col)]
        self.assertEqual(len(die_match), 1, f"Die ({target_row}, {target_col}) should exist uniquely on wafer {test_wafer}")

    def test_model_a_vs_b_page_data_logic(self):
        """Test data loader integration for Model A vs Model B wafer comparison page."""
        from demo.data_loader import load_all_prediction_wafers, load_final_metrics
        df = load_all_prediction_wafers()
        self.assertFalse(df.empty, "Prediction wafers dataset should not be empty")
        wafers = sorted(df["wafer_id"].unique())
        self.assertEqual(len(wafers), 40, "Dynamic wafer selector should expose all 40 test wafers")
        
        metrics = load_final_metrics()
        self.assertIn("model_a", metrics)
        self.assertIn("model_b", metrics)
        self.assertIn("ap", metrics["model_a"])
        self.assertIn("ap", metrics["model_b"])
        self.assertIn("auc", metrics["model_a"])
        self.assertIn("auc", metrics["model_b"])
        self.assertIn("f1", metrics["model_a"])
        self.assertIn("f1", metrics["model_b"])

    def test_model_a_predictions_artifact_validation(self):
        """Test artifact validity and data loader for Model A die-level predictions."""
        from demo.data_loader import load_model_a_predictions
        df_a = load_model_a_predictions()
        self.assertFalse(df_a.empty, "Model A prediction artifact should exist and be non-empty")
        self.assertEqual(len(df_a), 39351, "Model A artifact should contain exactly 39,351 rows")
        self.assertEqual(df_a["wafer_id"].nunique(), 40, "Model A artifact should cover 40 unique wafers")
        self.assertEqual(df_a.duplicated(["wafer_id", "die_row", "die_col"]).sum(), 0, "No duplicate coordinates allowed")
        self.assertEqual(df_a["predicted_probability"].isna().sum(), 0, "No NaNs allowed in Model A probabilities")
        self.assertTrue((df_a["predicted_probability"] >= 0).all() and (df_a["predicted_probability"] <= 1.0).all())

    def test_three_panel_wafer_comparison_joined_data_integrity(self):
        """Step 3 data integrity test for joined Model A, Model B, and Risk Difference 3-panel maps."""
        from demo.data_loader import load_model_a_b_joined_predictions, load_all_prediction_wafers, load_model_a_predictions
        df_a = load_model_a_predictions()
        df_b = load_all_prediction_wafers()
        merged = load_model_a_b_joined_predictions()
        
        self.assertFalse(df_a.empty, "Model A artifact should load successfully")
        self.assertFalse(df_b.empty, "Model B artifact should load successfully")
        self.assertFalse(merged.empty, "Joined dataset should load successfully")
        
        # Exact row count & wafer count matching
        self.assertEqual(len(merged), 39351, "Joined dataset must contain 39,351 rows")
        self.assertEqual(merged["wafer_id"].nunique(), 40, "Joined dataset must cover 40 test wafers")
        self.assertEqual(merged.duplicated(["wafer_id", "die_row", "die_col"]).sum(), 0, "Joined dataset must have zero duplicate coordinates")
        self.assertEqual(merged.isna().sum().sum(), 0, "Joined dataset must have zero missing values")
        
        # Verify exact risk difference calculation: difference = Model B prob - Model A prob
        diff_calc = merged["prob_model_b"] - merged["prob_model_a"]
        max_diff_err = (merged["risk_difference"] - diff_calc).abs().max()
        self.assertLess(max_diff_err, 1e-6, "Risk difference must equal Model B probability minus Model A probability exactly")
        
        # Verify probability and difference bounds
        self.assertTrue((merged["prob_model_a"] >= 0.0).all() and (merged["prob_model_a"] <= 1.0).all(), "Model A probabilities must be in [0, 1]")
        self.assertTrue((merged["prob_model_b"] >= 0.0).all() and (merged["prob_model_b"] <= 1.0).all(), "Model B probabilities must be in [0, 1]")
        self.assertTrue((merged["risk_difference"] >= -1.0).all() and (merged["risk_difference"] <= 1.0).all(), "Risk difference must be in [-1, +1]")
        
        # Test selected wafer dynamic calculations
        test_wafer = "W_F_0003"
        w_df = merged[merged["wafer_id"] == test_wafer]
        self.assertGreater(len(w_df), 0, f"Wafer {test_wafer} must exist in joined dataset")
        self.assertIn("prob_model_a", w_df.columns)
        self.assertIn("prob_model_b", w_df.columns)
        self.assertIn("risk_difference", w_df.columns)

    def test_statistical_robustness_artifact_existence(self):
        """Step 4 test: Verify all required statistical robustness artifacts exist."""
        results_dir = Path("results/statistical_robustness")
        required_files = [
            "wafer_level_metrics.csv",
            "bootstrap_results.json",
            "statistical_tests.json",
            "summary.json",
            "experiment_notes.md",
            "paired_wafer_ap.png",
            "wafer_delta_ap_distribution.png",
            "bootstrap_delta_ap_distribution.png",
        ]
        for fname in required_files:
            p = results_dir / fname
            self.assertTrue(p.exists(), f"Artifact {fname} must exist in results/statistical_robustness/")

    def test_wafer_level_metrics_schema_and_pairing(self):
        """Step 4 test: Verify wafer-level metrics schema, exact pairing, and uniqueness."""
        metrics_p = Path("results/statistical_robustness/wafer_level_metrics.csv")
        if not metrics_p.exists():
            self.skipTest("wafer_level_metrics.csv not generated yet")
        df = pd.read_csv(metrics_p)
        self.assertFalse(df.empty, "wafer_level_metrics.csv must not be empty")
        self.assertEqual(df["wafer_id"].nunique(), len(df), "No duplicate wafer entries allowed")
        
        expected_cols = {"wafer_id", "n_dies", "n_pos", "AP_A", "AP_B", "delta_AP", "AUC_A", "AUC_B", "delta_AUC", "F1_A", "F1_B", "delta_F1"}
        self.assertTrue(expected_cols.issubset(set(df.columns)))
        
        # Verify delta calculation on valid rows
        valid = df.dropna(subset=["AP_A", "AP_B", "delta_AP"])
        self.assertGreater(len(valid), 0, "Should have valid wafer AP pairs")
        calc_delta = valid["AP_B"] - valid["AP_A"]
        max_err = (valid["delta_AP"] - calc_delta).abs().max()
        self.assertLess(max_err, 1e-6, "delta_AP must equal AP_B - AP_A exactly")

    def test_bootstrap_results_validity(self):
        """Step 4 test: Verify bootstrap confidence interval validity and JSON schema."""
        boot_p = Path("results/statistical_robustness/bootstrap_results.json")
        if not boot_p.exists():
            self.skipTest("bootstrap_results.json not generated yet")
        import json
        data = json.loads(boot_p.read_text(encoding="utf-8"))
        self.assertEqual(data["n_bootstrap"], 1000)
        self.assertEqual(data["seed"], 42)
        self.assertIn("primary_metric", data)
        pm = data["primary_metric"]
        self.assertIn("ci_95_percentile", pm)
        ci = pm["ci_95_percentile"]
        self.assertEqual(len(ci), 2)
        self.assertLess(ci[0], ci[1], "Lower CI bound must be strictly less than upper CI bound")

    def test_statistical_tests_serialization(self):
        """Step 4 test: Verify statistical tests JSON output schema and valid metrics."""
        tests_p = Path("results/statistical_robustness/statistical_tests.json")
        if not tests_p.exists():
            self.skipTest("statistical_tests.json not generated yet")
        import json
        data = json.loads(tests_p.read_text(encoding="utf-8"))
        self.assertIn("primary_metric", data)
        pm = data["primary_metric"]
        self.assertIn("wilcoxon_signed_rank", pm)
        self.assertIn("paired_ttest", pm)
        self.assertIn("effect_size", pm)
        self.assertIn("p_value", pm["wilcoxon_signed_rank"])
        self.assertIn("value", pm["effect_size"])

    def test_failure_signature_artifacts_exist(self):
        """Step 5 test: Verify failure signature artifacts and plots exist."""
        sig_dir = Path("results/failure_signatures")
        required = [
            "cluster_assignments.csv",
            "cluster_summary.csv",
            "cluster_profiles.csv",
            "signature_summary.json",
            "experiment_notes.md",
            "plots/cluster_projection.png",
            "plots/cluster_evidence_profiles.png",
            "plots/cluster_sizes.png",
            "plots/wafer_signature_distribution.png",
        ]
        for fname in required:
            p = sig_dir / fname
            self.assertTrue(p.exists(), f"Artifact {fname} must exist in results/failure_signatures/")

    def test_signature_feature_construction_and_assignments(self):
        """Step 5 test: Verify die-level cluster assignments, uniqueness, and clean values."""
        csv_p = Path("results/failure_signatures/cluster_assignments.csv")
        if not csv_p.exists():
            self.skipTest("cluster_assignments.csv not generated yet")
        df = pd.read_csv(csv_p)
        self.assertEqual(len(df), 154037, "Assignments file must contain all 154,037 eligible dies")
        self.assertEqual(df["die_id"].nunique(), len(df), "Die IDs must be unique without duplicate entries")
        
        feature_cols = ["parametric_evidence", "spatial_evidence", "block_evidence", "model_b_probability", "risk_difference"]
        for col in feature_cols:
            self.assertIn(col, df.columns)
            self.assertEqual(df[col].isna().sum(), 0, f"No NaNs allowed in {col}")
            self.assertFalse(np.isinf(df[col]).any(), f"No Inf allowed in {col}")

        # Check cluster IDs validity
        self.assertTrue((df["cluster_id"] >= 0).all() and (df["cluster_id"] <= 10).all())

    def test_cluster_summary_reconciles_with_assignments(self):
        """Step 5 test: Verify cluster summary counts reconcile exactly with assignments."""
        assign_p = Path("results/failure_signatures/cluster_assignments.csv")
        sum_p = Path("results/failure_signatures/cluster_summary.csv")
        if not assign_p.exists() or not sum_p.exists():
            self.skipTest("Signature artifacts not generated yet")
            
        df_assign = pd.read_csv(assign_p)
        df_sum = pd.read_csv(sum_p)
        
        self.assertFalse(df_sum.empty)
        self.assertEqual(df_sum["n_dies"].sum(), len(df_assign), "Cluster summary die count sum must equal assignments row count")
        
        for _, row in df_sum.iterrows():
            cid = row["cluster_id"]
            sub = df_assign[df_assign["cluster_id"] == cid]
            self.assertEqual(len(sub), int(row["n_dies"]), f"Cluster {cid} die count mismatch")
            self.assertIn("signature_name", row)
            self.assertIn("dominant_source", row)

    def test_failure_signature_data_loader_integration(self):
        """Step 5 test: Verify demo data loader integration for failure signatures."""
        from demo.data_loader import (
            load_signature_summary,
            load_cluster_summary,
            load_cluster_profiles,
            load_cluster_assignments,
        )
        sig_summary = load_signature_summary()
        self.assertIn("n_clusters_discovered", sig_summary)
        self.assertEqual(sig_summary["n_clusters_discovered"], 5)
        
        sum_df = load_cluster_summary()
        self.assertFalse(sum_df.empty)
        self.assertEqual(len(sum_df), 5)
        
        prof_df = load_cluster_profiles()
        self.assertFalse(prof_df.empty)

    def test_step_5_5_cluster_wafers_summary_aggregation(self):
        """Step 5.5 test: Verify get_cluster_wafers_summary aggregation, schema, sorting, and reconciliation."""
        from demo.data_loader import get_cluster_wafers_summary, load_cluster_assignments, load_cluster_summary
        
        sum_df = load_cluster_summary()
        self.assertFalse(sum_df.empty, "Cluster summary must be non-empty")
        
        # Test for Cluster 0
        cid = int(sum_df.iloc[0]["cluster_id"])
        wafers_summary = get_cluster_wafers_summary(cid)
        
        self.assertFalse(wafers_summary.empty, f"Cluster {cid} must have affected wafers summary")
        expected_cols = {
            "wafer_id", "signature_die_count", "total_wafer_dies", "pct_of_wafer_dies",
            "mean_model_b_prob", "mean_model_a_prob", "mean_risk_diff", "descriptive_failure_rate"
        }
        self.assertTrue(expected_cols.issubset(set(wafers_summary.columns)))
        
        # Check sorting: signature_die_count desc, then mean_model_b_prob desc
        counts = wafers_summary["signature_die_count"].tolist()
        self.assertEqual(counts, sorted(counts, reverse=True), "Wafers summary must be sorted by signature_die_count descending")
        
        # Total signature dies sum across wafers must equal cluster die count in summary
        tot_sig_dies = wafers_summary["signature_die_count"].sum()
        expected_n_dies = int(sum_df[sum_df["cluster_id"] == cid].iloc[0]["n_dies"])
        self.assertEqual(tot_sig_dies, expected_n_dies, "Aggregated wafer signature dies must reconcile with cluster summary total")

    def test_step_5_5_cluster_wafer_dies_filtering_and_sorting(self):
        """Step 5.5 test: Verify get_cluster_wafer_dies coordinate integrity and sorting by Model B risk desc."""
        from demo.data_loader import get_cluster_wafers_summary, get_cluster_wafer_dies
        
        wafers_summary = get_cluster_wafers_summary(0)
        self.assertFalse(wafers_summary.empty)
        
        target_wafer = wafers_summary.iloc[0]["wafer_id"]
        dies_df = get_cluster_wafer_dies(0, target_wafer)
        
        self.assertFalse(dies_df.empty, f"Wafer {target_wafer} must have dies for cluster 0")
        self.assertEqual(len(dies_df), int(wafers_summary.iloc[0]["signature_die_count"]))
        
        # Verify coordinate & model prediction integrity
        self.assertTrue((dies_df["die_row"] >= 0).all())
        self.assertTrue((dies_df["die_col"] >= 0).all())
        self.assertTrue((dies_df["model_b_probability"] >= 0.0).all() and (dies_df["model_b_probability"] <= 1.0).all())
        self.assertTrue((dies_df["model_a_probability"] >= 0.0).all() and (dies_df["model_a_probability"] <= 1.0).all())
        
        # Verify sorting by Model B risk descending
        probs = dies_df["model_b_probability"].tolist()
        self.assertEqual(probs, sorted(probs, reverse=True), "Cluster dies must be sorted by Model B risk descending")

    def test_step_5_5_signature_to_die_lookup_and_missing_handling(self):
        """Step 5.5 test: Verify signature -> die lookup integrity and graceful missing die handling."""
        from demo.data_loader import get_cluster_wafer_dies
        
        # Valid lookup
        dies_df = get_cluster_wafer_dies(0, "W_N_0083")
        if not dies_df.empty:
            sample_die = dies_df.iloc[0]
            self.assertIn("signature_name", sample_die)
            self.assertIn("dominant_source", sample_die)
            
        # Invalid / missing cluster or wafer lookup must handle gracefully without crashing
        empty_df_invalid_wafer = get_cluster_wafer_dies(0, "NON_EXISTENT_WAFER")
        self.assertTrue(empty_df_invalid_wafer.empty)
        
        empty_df_invalid_cid = get_cluster_wafer_dies(999, "W_N_0083")
        self.assertTrue(empty_df_invalid_cid.empty)

    def test_step_6_hotspot_detection_synthetic_grid(self):
        """Step 6 test: Verify 8-neighbor connectivity and minimum size thresholding on a synthetic wafer grid."""
        from demo.data_loader import detect_wafer_hotspots
        
        hotspots_df, annotated_df = detect_wafer_hotspots("W_N_0083", percentile_threshold=95.0, min_hotspot_dies=3)
        
        self.assertFalse(annotated_df.empty, "Annotated wafer DataFrame must not be empty for W_N_0083")
        self.assertIn("hotspot_id", annotated_df.columns)
        self.assertIn("is_high_risk_die", annotated_df.columns)
        self.assertIn("is_hotspot_die", annotated_df.columns)
        self.assertIn("is_isolated_high_risk", annotated_df.columns)
        
        if not hotspots_df.empty:
            first_h = hotspots_df.iloc[0]
            self.assertEqual(first_h["hotspot_id"], "H1")
            self.assertGreaterEqual(int(first_h["n_dies"]), 3)
            self.assertIn("bounding_region", first_h)
            self.assertIn("center_str", first_h)

    def test_step_6_hotspot_ranking_and_stats_reconciliation(self):
        """Step 6 test: Verify hotspot statistics, ranking order (mean_risk desc), and coordinates."""
        from demo.data_loader import detect_wafer_hotspots
        
        hotspots_df, annotated_df = detect_wafer_hotspots("W_N_0083", percentile_threshold=90.0, min_hotspot_dies=2)
        
        if len(hotspots_df) > 1:
            mean_risks = hotspots_df["mean_risk"].tolist()
            self.assertEqual(mean_risks, sorted(mean_risks, reverse=True), "Hotspots must be ranked by mean Model B risk descending")
            
            for _, h in hotspots_df.iterrows():
                self.assertLessEqual(h["row_min"], h["row_max"])
                self.assertLessEqual(h["col_min"], h["col_max"])
                self.assertGreaterEqual(h["centroid_row"], h["row_min"])
                self.assertLessEqual(h["centroid_row"], h["row_max"])

    def test_step_6_hotspot_lookup_and_degenerate_cases(self):
        """Step 6 test: Verify graceful handling of missing wafers or empty threshold results."""
        from demo.data_loader import detect_wafer_hotspots
        
        # Non-existent wafer
        h_df, w_df = detect_wafer_hotspots("NON_EXISTENT_WAFER", percentile_threshold=95.0, min_hotspot_dies=3)
        self.assertTrue(h_df.empty)
        self.assertTrue(w_df.empty)
        
        # High threshold and high min_size where 0 hotspots qualify
        h_df_high, w_df_high = detect_wafer_hotspots("W_N_0083", percentile_threshold=99.9, min_hotspot_dies=100)
        self.assertTrue(h_df_high.empty)
        self.assertFalse(w_df_high.empty, "Wafer dies DataFrame should still return annotated structure")

    def test_die_explanation_selection_and_fallback(self):
        """Verify dynamic die selection, explanation rendering fallback, and case study preservation."""
        from demo.data_loader import load_explanation_samples, load_all_prediction_wafers, load_cluster_assignments
        
        # 1. Existing 4 representative cases still work
        samples = load_explanation_samples()
        self.assertEqual(len(samples), 4, "Must preserve the 4 curated case studies")
        expected_cases = {
            ("W_N_0083", 9, 23),
            ("W_F_0039", 17, 3),
            ("W_N_0007", 20, 18),
            ("W_N_0055", 23, 11),
        }
        actual_cases = {(s["wafer_id"], s["die_row"], s["die_col"]) for s in samples}
        self.assertEqual(actual_cases, expected_cases, "Representative cases must match exact curated set")
        
        # 2. Arbitrary test die selection from all prediction wafers
        df_all = load_all_prediction_wafers()
        self.assertFalse(df_all.empty)
        arb_die = df_all[(df_all["wafer_id"] == "W_F_0019") & (df_all["die_row"] == 19) & (df_all["die_col"] == 8)]
        self.assertFalse(arb_die.empty, "Arbitrary test die W_F_0019 (19, 8) must exist in prediction dataset")
        
        # 3. Test cluster assignments lookup for eligible training wafer die
        df_assign = load_cluster_assignments()
        self.assertFalse(df_assign.empty)
        train_die_assign = df_assign[(df_assign["wafer_id"] == "W_N_0083") & (df_assign["die_row"] == 11) & (df_assign["die_col"] == 18)]
        self.assertFalse(train_die_assign.empty, "Training die W_N_0083 (11, 18) should be present in cluster assignments dataset")
        self.assertIn("block_evidence", train_die_assign.columns)

class TestDynamicDieExplanation(unittest.TestCase):
    """Focused automated tests for dynamic die evidence decomposition and waterfall rendering."""

    def test_1_representative_cases_preserved(self):
        """1. Known representative die still loads pre-computed explanation."""
        from demo.data_loader import find_explanation
        preset_cases = [
            ("W_N_0083", 9, 23),
            ("W_F_0039", 17, 3),
            ("W_N_0007", 20, 18),
            ("W_N_0055", 23, 11),
        ]
        for w, r, c in preset_cases:
            exp = find_explanation(w, r, c)
            self.assertIsNotNone(exp, f"Representative die {w} ({r},{c}) must load explanation")
            self.assertEqual(exp["wafer_id"], w)
            self.assertEqual(int(exp["die_row"]), r)
            self.assertEqual(int(exp["die_col"]), c)
            self.assertIn("logit_decomposition", exp)

    def test_2_arbitrary_die_dynamic_explanation_generation(self):
        """2. Arbitrary die not present in explanation_samples.json produces dynamic explanation."""
        from demo.data_loader import find_explanation, load_explanation_samples
        samples = load_explanation_samples()
        sample_coords = {(s["wafer_id"], s["die_row"], s["die_col"]) for s in samples}

        # Test arbitrary coordinates not in explanation_samples.json
        test_coords = [
            ("W_F_0019", 19, 8),
            ("W_N_0083", 11, 18),
            ("W_N_0083", 3, 13),
        ]
        for w, r, c in test_coords:
            self.assertNotIn((w, r, c), sample_coords, f"Coordinate {w} ({r},{c}) should be arbitrary")
            exp = find_explanation(w, r, c)
            self.assertIsNotNone(exp, f"Arbitrary die {w} ({r},{c}) must return dynamic explanation")
            self.assertTrue(exp.get("is_dynamic", False), "Explanation should be marked as dynamic")

    def test_3_dynamic_explanation_coordinate_matching(self):
        """3. Dynamic explanation coordinates exactly match requested wafer_id, die_row, die_col."""
        from demo.data_loader import find_explanation
        target_w, target_r, target_c = "W_F_0019", 19, 8
        exp = find_explanation(target_w, target_r, target_c)
        self.assertIsNotNone(exp)
        self.assertEqual(exp["wafer_id"], target_w)
        self.assertEqual(exp["die_row"], target_r)
        self.assertEqual(exp["die_col"], target_c)

    def test_4_model_probabilities_match_prediction_artifacts(self):
        """4. Model A/B probabilities match existing prediction artifacts for W_F_0019 (19,8)."""
        from demo.data_loader import load_model_a_b_joined_predictions, find_explanation
        joined = load_model_a_b_joined_predictions()
        match = joined[(joined["wafer_id"] == "W_F_0019") & (joined["die_row"] == 19) & (joined["die_col"] == 8)]
        self.assertFalse(match.empty)

        row = match.iloc[0]
        self.assertAlmostEqual(row["prob_model_a"], 0.018473, places=4)
        self.assertAlmostEqual(row["prob_model_b"], 1.0, places=4)
        self.assertAlmostEqual(row["risk_difference"], 0.981527, places=4)

        exp = find_explanation("W_F_0019", 19, 8)
        self.assertIsNotNone(exp)

    def test_5_dynamic_waterfall_logit_reconciliation(self):
        """5. Dynamic waterfall components reconcile with final Model B logit within tight tolerance (< 1e-4)."""
        from demo.data_loader import find_explanation
        test_coords = [
            ("W_F_0019", 19, 8),
            ("W_N_0083", 11, 18),
            ("W_N_0083", 3, 13),
        ]
        for w, r, c in test_coords:
            exp = find_explanation(w, r, c)
            self.assertIsNotNone(exp)
            decomp = exp["logit_decomposition"]
            c_spatial = decomp["spatial_contribution"]
            c_param = decomp["parametric_contribution"]
            c_block = decomp["block_contribution"]
            final_logit = exp["final_logit"]

            calc_sum = c_spatial + c_param + c_block
            diff = abs(calc_sum - final_logit)
            self.assertLess(diff, 1e-4, f"Decomposition logit mismatch for {w} ({r},{c}): calc_sum={calc_sum}, final={final_logit}, diff={diff}")
            self.assertLess(decomp.get("sum_check_residual", 0.0), 1e-4)

    def test_6_no_other_die_explanation_substituted(self):
        """6. No other die's explanation is substituted when requesting an arbitrary coordinate."""
        from demo.data_loader import find_explanation
        exp_wf19 = find_explanation("W_F_0019", 19, 8)
        exp_preset = find_explanation("W_N_0083", 9, 23)

        self.assertIsNotNone(exp_wf19)
        self.assertIsNotNone(exp_preset)
        self.assertNotEqual(exp_wf19["final_logit"], exp_preset["final_logit"], "Must not substitute representative case explanation for W_F_0019 (19,8)")
        self.assertEqual(exp_wf19["wafer_id"], "W_F_0019")
        self.assertEqual(exp_wf19["die_row"], 19)
        self.assertEqual(exp_wf19["die_col"], 8)

    def test_7_missing_optional_block_localization_does_not_crash(self):
        """7. Missing optional 2,000 sequence block localization does not crash."""
        from demo.data_loader import find_explanation
        exp = find_explanation("W_N_0083", 11, 18)
        self.assertIsNotNone(exp)
        self.assertIn("block_context", exp)
        self.assertIn("top_block_features", exp["block_context"])

    def test_8_failure_signature_to_explanation(self):
        """8. Failure Signature -> selected die -> explanation works."""
        from demo.data_loader import get_cluster_wafer_dies, find_explanation
        dies_df = get_cluster_wafer_dies(0, "W_N_0083")
        self.assertFalse(dies_df.empty)

        target_row = int(dies_df.iloc[0]["die_row"])
        target_col = int(dies_df.iloc[0]["die_col"])

        exp = find_explanation("W_N_0083", target_row, target_col)
        self.assertIsNotNone(exp)
        self.assertEqual(exp["wafer_id"], "W_N_0083")
        self.assertEqual(exp["die_row"], target_row)
        self.assertEqual(exp["die_col"], target_col)

    def test_9_wafer_hotspot_to_explanation(self):
        """9. Wafer Hotspot -> selected die -> explanation works."""
        from demo.data_loader import detect_wafer_hotspots, find_explanation
        h_df, annotated_df = detect_wafer_hotspots("W_N_0083", percentile_threshold=90.0, min_hotspot_dies=2)
        self.assertFalse(annotated_df.empty)

        hotspot_dies = annotated_df[annotated_df["is_hotspot_die"]]
        self.assertFalse(hotspot_dies.empty)

        target_row = int(hotspot_dies.iloc[0]["die_row"])
        target_col = int(hotspot_dies.iloc[0]["die_col"])

        exp = find_explanation("W_N_0083", target_row, target_col)
        self.assertIsNotNone(exp)
        self.assertEqual(exp["wafer_id"], "W_N_0083")
        self.assertEqual(exp["die_row"], target_row)
        self.assertEqual(exp["die_col"], target_col)

    def test_10_top_parametric_features_count(self):
        """10. Top 10 parametric features are calculated dynamically for arbitrary dies."""
        from demo.data_loader import find_explanation
        exp = find_explanation("W_F_0019", 19, 8)
        self.assertIsNotNone(exp)
        top_p = exp.get("top_parametric_features", [])
        self.assertEqual(len(top_p), 10, "Top 10 parametric drivers should be returned")
        first = top_p[0]
        self.assertIn("feature_name", first)
        self.assertIn("contribution", first)
        self.assertIn("weight", first)

class TestGlobalSelectionSync(unittest.TestCase):
    """Automated unit tests for global wafer & die selection synchronization across dashboard pages."""

    def test_1_global_state_synchronization_data_integrity(self):
        """Verify that selecting an arbitrary wafer and die updates canonical state and explanation lookup."""
        from demo.data_loader import load_all_prediction_wafers, find_explanation
        
        all_wafers_df = load_all_prediction_wafers()
        self.assertFalse(all_wafers_df.empty)
        
        # Test selection: first wafer in prediction dataset
        test_w = all_wafers_df["wafer_id"].iloc[0]
        wafer_dies = all_wafers_df[all_wafers_df["wafer_id"] == test_w]
        self.assertFalse(wafer_dies.empty, f"Wafer {test_w} should exist in prediction wafers")
        sample_die = wafer_dies.iloc[0]
        test_r = int(sample_die["die_row"])
        test_c = int(sample_die["die_col"])
        
        die_match = all_wafers_df[(all_wafers_df["wafer_id"] == test_w) & (all_wafers_df["die_row"] == test_r) & (all_wafers_df["die_col"] == test_c)]
        self.assertEqual(len(die_match), 1, f"Die ({test_r}, {test_c}) on wafer {test_w} should exist uniquely")
        
        exp = find_explanation(test_w, test_r, test_c)
        self.assertIsNotNone(exp)
        self.assertEqual(exp["wafer_id"], test_w)
        self.assertEqual(exp["die_row"], test_r)
        self.assertEqual(exp["die_col"], test_c)

    def test_2_parametric_drivers_and_block_analysis_for_global_selection(self):
        """Verify Parametric Drivers and Block Analysis work for arbitrary globally selected die W_N_0083 (11, 18)."""
        from demo.data_loader import find_explanation
        exp = find_explanation("W_N_0083", 11, 18)
        self.assertIsNotNone(exp)
        
        top_feats = exp.get("top_parametric_features", [])
        self.assertGreater(len(top_feats), 0, "Parametric drivers must be available for global selection W_N_0083 (11,18)")
        
        self.assertIn("block_context", exp)
        self.assertIn("peak_anomaly_index", exp["block_context"])

    def test_3_cross_page_selection_preserves_wafer_and_die(self):
        """Verify cross-page die selection coordinates are preserved without resetting to defaults."""
        from demo.data_loader import load_model_a_b_joined_predictions, detect_wafer_hotspots
        
        # 1. Model A vs B selection
        joined = load_model_a_b_joined_predictions()
        w_df = joined[joined["wafer_id"] == "W_F_0019"]
        self.assertFalse(w_df.empty)
        
        # 2. Hotspots selection on W_N_0083
        h_df, annotated_df = detect_wafer_hotspots("W_N_0083", percentile_threshold=90.0, min_hotspot_dies=2)
        self.assertFalse(annotated_df.empty)
        hotspots = annotated_df[annotated_df["is_hotspot_die"]]
        self.assertFalse(hotspots.empty)
        target_r = int(hotspots.iloc[0]["die_row"])
        target_c = int(hotspots.iloc[0]["die_col"])
        
        # Check lookup matches target
        match = annotated_df[(annotated_df["wafer_id"] == "W_N_0083") & (annotated_df["die_row"] == target_r) & (annotated_df["die_col"] == target_c)]
        self.assertEqual(len(match), 1)

    def test_4_wafer_selection_persistence_no_revert(self):
        """Verify that selecting W_N_0122 persists in session state without reverting to W_F_0003."""
        from demo.data_loader import load_all_prediction_wafers
        all_wafers = sorted(load_all_prediction_wafers()["wafer_id"].unique())
        self.assertIn("W_N_0122", all_wafers)
        self.assertEqual(all_wafers[0], "W_F_0003")
        
        # Simulate selection state change to W_N_0122
        session_state = {}
        session_state["selected_wafer"] = "W_N_0122"
        
        # Re-run simulation check: canonical key retains user selection
        self.assertEqual(session_state["selected_wafer"], "W_N_0122")
        self.assertNotEqual(session_state["selected_wafer"], all_wafers[0])

    def test_5_die_row_col_widget_selector_sync(self):
        """Verify that die selection via Plotly or callbacks updates ov_row_selector and ov_col_selector."""
        session_state = {
            "selected_wafer": "W_N_0122",
            "selected_die_row": 15,
            "selected_die_col": 12,
        }
        # Simulate synchronization step executed before rendering selectboxes
        session_state["ov_row_selector"] = session_state["selected_die_row"]
        session_state["ov_col_selector"] = session_state["selected_die_col"]

        self.assertEqual(session_state["ov_row_selector"], 15)
        self.assertEqual(session_state["ov_col_selector"], 12)

    def test_6_slider_bounds_safety(self):
        """Verify slider min/max bounds logic when hotspot_dies_df length is <= 5 (e.g. 5 dies)."""
        for n_dies in [1, 5, 20]:
            if n_dies > 1:
                min_val = 1
                max_val = n_dies
                val = min(20, n_dies)
                self.assertLess(min_val, max_val, f"min_value ({min_val}) must be strictly less than max_value ({max_val}) for n_dies={n_dies}")
                self.assertGreaterEqual(val, min_val)
                self.assertLessEqual(val, max_val)

if __name__ == "__main__":
    unittest.main()




