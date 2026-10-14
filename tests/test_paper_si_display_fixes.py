"""Presentation-only regression checks for S10 boundary marks and S12 keys."""

import numpy as np
import pandas as pd
from matplotlib.markers import MarkerStyle

from paper.figures import figstyle
from paper.figures import figS10_fraction_reliability as s10
from paper.figures import figS12_retransformation as s12


class RegistryStub:
    def __init__(self, frames):
        self.frames = frames
        self.payload = {"tables": {
            name: {"source_file": name + ".csv", "source_sha256": "a" * 64}
            for name in frames
        }}

    def table(self, name, **filters):
        frame = self.frames[name]
        for key, value in filters.items():
            frame = frame.loc[frame[key].eq(value)]
        return frame.copy()


def marker_vertices(marker):
    style = MarkerStyle(marker)
    return style.get_path().transformed(style.get_transform()).vertices


def test_s12_branch_shape_smearing_fill_and_legend_preserve_data(monkeypatch, tmp_path):
    rows = [dict(reference=reference, branch=branch, population="own",
                 scheme=scheme, repeat=0, split_id="fold0", training_rows=100,
                 evaluation_positive_rows=2, positive_observed_mean=.5,
                 f_pos_mean=.6, f_pos_star_mean=.55,
                 selection_criterion="site_month_RMSE",
                 target="dust fraction (NOT FMF)",
                 coefficient_order="a + b*DB_AE + c*DB_AE^2", a=.4, b=-.2, c=.1)
            for reference in s12.REFERENCES for branch in s12.BRANCHES
            for scheme in ("site_cv", "region_out")]
    registry = RegistryStub({s12.ROOT + name: pd.DataFrame(rows) for name in s12.TABLES})
    expected = s12.select(registry)[1]
    monkeypatch.setattr(s12.Registry, "read", lambda path: registry)
    captured = []

    def export(fig, output, stem, data, *args):
        pd.testing.assert_frame_equal(data, expected)
        for ax in fig.axes[:2]:
            assert ax.get_xlim() == (0, 1)
            assert ax.get_ylim() == (0, 1)
            assert len(ax.collections) == 4
            for index, collection in enumerate(ax.collections):
                marker = "o" if index < 2 else "s"
                np.testing.assert_array_equal(collection.get_paths()[0].vertices,
                                              marker_vertices(marker))
                smeared = index % 2 == 1
                assert bool(len(collection.get_facecolors())) == smeared
                np.testing.assert_array_equal(collection.get_offsets(),
                                              [[.5, .55 if smeared else .6]] * 2)
        for ax in fig.axes[2:]:
            assert len(ax.collections) == 6
            for index, collection in enumerate(ax.collections):
                branch_index, coefficient_index = divmod(index, 3)
                np.testing.assert_array_equal(collection.get_paths()[0].vertices,
                                              marker_vertices("o" if branch_index == 0 else "s"))
                # Existing display offsets and saved coefficients must not move.
                x = coefficient_index + (branch_index - .5) * .20 + np.linspace(-.055, .055, 2)
                np.testing.assert_array_equal(collection.get_offsets()[:, 0], x)
                np.testing.assert_array_equal(collection.get_offsets()[:, 1],
                                              [[.4, -.2, .1][coefficient_index]] * 2)
        legend = fig.legends[0]
        handles = legend.legend_handles
        assert [h.get_marker() for h in handles] == ["o", "s", "o", "o"]
        assert [h.get_markerfacecolor() for h in handles] == ["none", "none", "none", "black"]
        assert [t.get_text() for t in legend.get_texts()] == [
            "full8: own population", "noSSA5: own population",
            "Unsmeared positive mean", "Residual-smeared positive mean"]
        captured.append(stem)

    monkeypatch.setattr(figstyle, "export", export)
    s12.make(tmp_path / "registry.json", tmp_path)
    assert captured == ["FigS12_retransformation"]
    caption = (tmp_path / "FigS12_caption.md").read_text()
    assert "Circles denote full8 and squares noSSA5 in both rows" in caption
    assert "open markers" in caption and "filled markers" in caption
    assert "filled triangles" not in caption


def test_s10_boundary_markers_unclipped_above_spines_without_data_changes(monkeypatch, tmp_path):
    name = "expected_value_test/stage_b/calibration_bins"
    rows = [dict(stage="b", reference="experimental1020", branch=branch,
                 estimator=method, scheme=scheme, bin_low=value, bin_high=value + .1,
                 predicted=value, observed=value, observed_lo=value, observed_hi=value,
                 sites=2 if value == 0 else 8)
            for branch in ("full8_common", "noSSA5_own", "hierarchy_own")
            for method in ("Eselected", "E1", "E5")
            for scheme in ("site_cv", "region_out") for value in (0., 1.)]
    expected = pd.DataFrame(rows)
    registry = RegistryStub({name: expected})
    monkeypatch.setattr(s10.Registry, "read", lambda path: registry)
    captured = []

    def export(fig, output, stem, data, *args):
        pd.testing.assert_frame_equal(data, expected)
        assert len(fig.axes) == 6
        for ax in fig.axes:
            assert ax.get_xlim() == (0, 1) and ax.get_ylim() == (0, 1)
            np.testing.assert_array_equal(ax.get_xticks(), [0, .5, 1])
            np.testing.assert_array_equal(ax.get_yticks(), [0, .5, 1])
            markers = [line for line in ax.lines if line.get_marker() in ("o", "s", "D")]
            assert len(markers) == 6
            for index, line in enumerate(markers):
                value = 1. if index % 2 == 0 else 0.
                np.testing.assert_array_equal(line.get_xdata(), [value])
                np.testing.assert_array_equal(line.get_ydata(), [value])
                assert not line.get_clip_on()
                assert line.get_zorder() > max(spine.get_zorder() for spine in ax.spines.values())
                assert (line.get_markerfacecolor() == "white") == (value == 0)
            assert len(ax.collections) == 3
            for intervals in ax.collections:
                assert intervals.get_clip_on()
                np.testing.assert_array_equal(intervals.get_segments(),
                                              [[[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]]])
        captured.append(stem)

    monkeypatch.setattr(figstyle, "export", export)
    s10.make(tmp_path / "registry.json", tmp_path)
    assert captured == ["FigS10_fraction_reliability"]
