from __future__ import annotations
import numpy as np

from audiopipe.score import compile_placements, Placement
from audiopipe.mix import mix_placements, pan_gains


def _voice(name, **kw):
    base = {"name": name, "content": f"{name}.wav", "offset": 0.0, "gain": 1.0,
            "pan": 0.0, "pitch": 0.0}
    return {**base, **kw}


# --- step 1: the phasing / compile math -------------------------------------

def test_period_placement_count_and_frames():
    # floor((D - offset)/P) + 1 placements at the expected frames
    pl = compile_placements([_voice("a", period=10.0)], duration=35.0, sr=100)
    assert len(pl) == 4                                   # 0,10,20,30
    assert [p.start_frame for p in pl] == [0, 1000, 2000, 3000]
    assert [p.cycle for p in pl] == [0, 1, 2, 3]


def test_offset_shifts_grid():
    pl = compile_placements([_voice("a", period=10.0, offset=3.0)], duration=35.0, sr=100)
    assert [p.start_frame for p in pl] == [300, 1300, 2300, 3300]   # 3,13,23,33


def test_period_and_events_merge_and_cycle_monotonic():
    v = _voice("a", period=10.0, events=[{"at": 5.0}, {"at": 25.0, "pitch": 3, "gain": 0.6}])
    pl = compile_placements([v], duration=35.0, sr=100)
    assert [p.start_frame / 100 for p in pl] == [0, 5, 10, 20, 25, 30]
    assert [p.cycle for p in pl] == [0, 1, 2, 3, 4, 5]   # monotonic across merged entries
    ev = next(p for p in pl if p.start_frame == 2500)
    assert ev.pitch == 3 and ev.gain == 0.6             # event overrides carried through


def test_voice_needs_period_or_events():
    import pytest
    with pytest.raises(ValueError, match="neither period nor events"):
        compile_placements([_voice("a")], duration=10.0, sr=100)


def test_incommensurate_periods_do_not_realign():
    # Eno's trick: 17.8 and 20.1 never coincide again within a long span
    D = 600.0
    a = [p.start_frame for p in compile_placements([_voice("a", period=17.8)], D, 1000)]
    b = set(p.start_frame for p in compile_placements([_voice("b", period=20.1)], D, 1000))
    coincide = [f for f in a if f != 0 and f in b]
    assert coincide == []                                # only align at t=0


# --- step 2: the mixer ------------------------------------------------------

def test_pan_is_equal_power():
    for pan in (-1.0, -0.3, 0.0, 0.6, 1.0):
        lg, rg = pan_gains(pan)
        assert abs(lg ** 2 + rg ** 2 - 1.0) < 1e-6      # constant power across the pan


def test_mix_lands_energy_at_exact_frames_and_pans():
    click = np.zeros((10, 1), dtype="float32"); click[0] = 1.0
    placements = [Placement("a", "x", 100, 0, 1.0, -1.0, 0.0),   # hard left @100
                  Placement("b", "x", 250, 0, 1.0, 1.0, 0.0)]    # hard right @250
    m = mix_placements(placements, {"a": click, "b": click}, 1000, normalize_db=-1.0)
    assert m.shape == (1000, 2)
    assert m[100, 0] != 0 and abs(m[100, 1]) < 1e-6     # left only at 100
    assert m[250, 1] != 0 and abs(m[250, 0]) < 1e-6     # right only at 250
    assert np.all(m[:100] == 0) and np.all(m[251:] == 0)


def test_mix_normalizes_to_ceiling():
    click = np.zeros((10, 1), dtype="float32"); click[0] = 0.9
    m = mix_placements([Placement("a", "x", 0, 0, 1.0, 0.0, 0.0)], {"a": click}, 100,
                       normalize_db=-1.0)
    assert abs(np.max(np.abs(m)) - 10 ** (-1.0 / 20)) < 1e-4   # peak == the ceiling


def test_overlap_sums():
    # two placements overlapping at the same frame add
    tone = np.ones((10, 1), dtype="float32") * 0.3
    m = mix_placements([Placement("a", "x", 0, 0, 1.0, 0.0, 0.0),
                        Placement("b", "x", 0, 0, 1.0, 0.0, 0.0)],
                       {"a": tone, "b": tone}, 10, normalize_db=0.0)
    # both centre-panned (0.707 each) and summed, then normalized to 0 dBFS -> peak 1.0
    assert abs(np.max(np.abs(m)) - 1.0) < 1e-4
