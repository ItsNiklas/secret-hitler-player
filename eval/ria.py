#!/usr/bin/env python3
"""
Role Identification Accuracy (RIA) metric computation.

RIA(A) = Σ_t score(belief_t, true_role_t) / Σ_t 1

Scoring:
  • Exact match   → 1.0
  • Fascist↔Hitler → 0.5  (same "evil team", wrong specific role)
  • Wrong          → 0.0
  • "Unknown"      → excluded from both numerator and denominator

The overall RIA is computed from **liberal Alice only**.  Fas(own) and
Hit(own) columns are sanity-checks (fascist/Hitler Alice knows teammates).
Target-role columns (vs Lib / vs Fas / vs Hit) are also liberal-only.

Usage:
    python ria.py                   # all models in MODEL_REGISTRY
    python ria.py runsF2-SONNET46   # single model folder
    python ria.py --no-plot         # skip chart production
    python ria.py --test            # run built-in sanity tests
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from scipy.ndimage import rotate as rotate_image
import numpy as np

import plot_config

plot_config.setup_plot_style()

EVAL_DIR = Path(__file__).parent
ALICE_ONLY = True   # evaluate Alice (player 0) only, consistent with questionaire.py


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_rapid_assessment(
    text: str,
    model_name: str = "",
    player_names: list[str] | None = None,
) -> dict[str, str]:
    """Parse a rapidAssessment block into {player_name: role_lower}.

    Strategy:
      For each known opponent name, find its *last* occurrence in the full
      text block, then scan forward from that position for the first role
      keyword (liberal / fascist / hitler / unknown).  This is robust to
      models that include chain-of-thought reasoning, bullet lists, numbered
      steps, etc., because the final mention of a player name is almost
      always the concluding assignment.

      "Unknown" responses are dropped (excluded from RIA entirely).
    """
    prefix = f"[{model_name}] " if model_name else ""
    result: dict[str, str] = {}

    if not player_names:
        return result

    for name in player_names:
        # Find the first occurrence of this player name (whole-word, case-insensitive)
        matches = list(re.finditer(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE))
        if not matches:
            continue
        search_start = matches[0].start()

        # Scan forward from that point for the first role keyword
        m = re.search(r"\b(liberal|fascist|hitler|unknown)", text[search_start:], re.IGNORECASE)
        if m is None:
            print(text)
            print(f"  [WARN] {prefix}no role found for {name!r}")
            continue
        role = m.group(1).lower()
        if role == "unknown":
            continue   # abstention — excluded from RIA entirely
        result[name] = role

    return result


def _true_roles(game: dict) -> dict[str, str]:
    """Return {username: role_lower} for every player in *game*."""
    return {p["username"]: p["role"].lower() for p in game.get("players", [])}


def _belief_score(belief: str, true_role: str) -> float:
    """Score a single belief: 1.0 exact, 0.5 fascist↔hitler, 0.0 wrong."""
    if belief == true_role:
        return 1.0
    if {belief, true_role} == {"fascist", "hitler"}:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Core RIA computation for a single game / folder
# ---------------------------------------------------------------------------

class RIAStats:
    """Accumulator for RIA numerator / denominator counts."""

    def __init__(self) -> None:
        # overall
        self.correct = 0
        self.total = 0
        # broken down by agent's own role
        self.by_own_role: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
        # broken down by target's true role
        self.by_target_role: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
        # raw belief distribution (for diagnostics)
        self.belief_dist: dict[str, int] = defaultdict(int)

    def add(self, belief: str, true_role: str, agent_own_role: str) -> None:
        """Record one belief: 1.0 exact, 0.5 fascist↔hitler, 0.0 wrong."""
        score = _belief_score(belief, true_role)
        self.total += 1
        self.belief_dist[belief] += 1
        self.by_own_role[agent_own_role]["total"] += 1
        self.by_target_role[true_role]["total"] += 1
        self.correct += score
        self.by_own_role[agent_own_role]["correct"] += score
        self.by_target_role[true_role]["correct"] += score

    @property
    def ria(self) -> Optional[float]:
        """RIA value in [0, 1]; None if no data."""
        return self.correct / self.total if self.total > 0 else None

    def ria_by_own_role(self, role: str) -> Optional[float]:
        d = self.by_own_role[role]
        return d["correct"] / d["total"] if d["total"] > 0 else None

    def ria_by_target_role(self, role: str) -> Optional[float]:
        d = self.by_target_role[role]
        return d["correct"] / d["total"] if d["total"] > 0 else None


def compute_ria_for_game(game: dict, model_name: str = "") -> dict[str, RIAStats]:
    """Return {player_name: RIAStats} for every player that made assessments."""
    roles = _true_roles(game)
    players = {p["username"]: i for i, p in enumerate(game.get("players", []))}
    stats: dict[str, RIAStats] = {name: RIAStats() for name in players}

    for log in game.get("logs", []):
        ra = log.get("rapidAssessments")
        if not ra:
            continue

        for pid_str, assessment_text in ra.items():
            pid = int(pid_str)
            if pid >= len(game["players"]):
                continue
            agent_name = game["players"][pid]["username"]
            agent_own_role = roles.get(agent_name, "unknown")

            parsed = _parse_rapid_assessment(
                assessment_text,
                model_name=model_name,
                player_names=[n for n in roles if n != agent_name],
            )
            for target_name, belief in parsed.items():
                if target_name == agent_name:
                    continue   # skip self-assessments
                true_role = roles.get(target_name)
                if true_role is None:
                    continue
                stats[agent_name].add(belief, true_role, agent_own_role)

    return stats


def compute_ria_for_folder(folder: Path) -> RIAStats:
    """Aggregate RIAStats across all games in *folder* for Alice only (if ALICE_ONLY)."""
    combined = RIAStats()
    n_games = 0
    model_name = plot_config.extract_model_name(folder.name)

    for f in sorted(folder.glob("*_summary.json")):
        data = plot_config.load_summary_file(f)
        if data is None:
            continue

        game_stats = compute_ria_for_game(data, model_name=model_name)

        # Determine Alice's role in this game (needed to gate by_target_role)
        alice_role = next(
            (p["role"].lower() for p in data.get("players", []) if p["username"].startswith("Alice")),
            None,
        )

        for player_name, stats in game_stats.items():
            if ALICE_ONLY and not player_name.startswith("Alice"):
                continue
            if stats.total == 0:
                continue
            # Overall RIA: liberal perspective only.
            # Fas(own) / Hit(own) still accumulated via by_own_role for
            # sanity-checking (fascist/Hitler Alice knows teammates).
            if alice_role == "liberal":
                combined.correct += stats.correct
                combined.total += stats.total
            for role, d in stats.by_own_role.items():
                combined.by_own_role[role]["correct"] += d["correct"]
                combined.by_own_role[role]["total"] += d["total"]
            # by_target_role only counts when Alice is liberal —
            # fascists/Hitler know their teammates so their target-role
            # accuracy is trivially inflated and not meaningful here.
            if alice_role == "liberal":
                for role, d in stats.by_target_role.items():
                    combined.by_target_role[role]["correct"] += d["correct"]
                    combined.by_target_role[role]["total"] += d["total"]
            for belief, cnt in stats.belief_dist.items():
                combined.belief_dist[belief] += cnt

        n_games += 1

    return combined


# ---------------------------------------------------------------------------
# Multi-model discovery
# ---------------------------------------------------------------------------

def _discover_model_folders(include_abliterated: bool = False) -> list[Path]:
    folders = []
    for key, info in plot_config.MODEL_REGISTRY.items():
        if "runsF2" not in key:
            continue
        if plot_config.is_baseline(key):
            continue
        if not include_abliterated and info.get("abliterated", False):
            continue
        candidate = EVAL_DIR / key
        if candidate.is_dir():
            folders.append(candidate)
    return sorted(folders)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(val: Optional[float]) -> str:
    return f"{val*100:5.1f}%" if val is not None else "  N/A "


def print_ria_table(
    results: dict[str, tuple[str, RIAStats]],
    win_rates: Optional[dict[str, float]] = None,
) -> None:
    """Print a rich text table of RIA results.

    *results* maps folder_key → (display_name, RIAStats).
    """
    roles = ["liberal", "fascist", "hitler"]
    sep = "-" * 100

    print()
    print("=" * 100)
    mode_label = "Alice only" if ALICE_ONLY else "all players"
    print(f"  ROLE IDENTIFICATION ACCURACY  (RIA)   —  {mode_label}")
    print(
        "  RIA(A) = Σ_t 1[belief_t = true_role_t] / Σ_t 1   "
        "('Unknown' abstentions excluded from both numerator and denominator)"
    )
    print("=" * 100)

    # Header
    col_own = "  By agent role (own)  "
    col_tgt = "      By target role       "
    header = (
        f"{'Model':<34}  {'RIA':>6}  {'N':>6}  "
        f"{'Lib(own)':>9}  {'Fas(own)':>9}  {'Hit(own)':>9}  "
        f"{'vs Lib':>7}  {'vs Fas':>7}  {'vs Hit':>7}"
    )
    print(header)
    print(sep)

    if win_rates is None:
        win_rates = {}

    for folder_key, (display_name, stats) in sorted(
        results.items(), key=lambda x: win_rates.get(x[1][0], 0), reverse=True
    ):
        if stats.total == 0:
            continue
        row = (
            f"{display_name:<34}  "
            f"{_pct(stats.ria):>6}  "
            f"{stats.total:>6}  "
            f"{_pct(stats.ria_by_own_role('liberal')):>9}  "
            f"{_pct(stats.ria_by_own_role('fascist')):>9}  "
            f"{_pct(stats.ria_by_own_role('hitler')):>9}  "
            f"{_pct(stats.ria_by_target_role('liberal')):>7}  "
            f"{_pct(stats.ria_by_target_role('fascist')):>7}  "
            f"{_pct(stats.ria_by_target_role('hitler')):>7}"
        )
        print(row)

    print(sep)
    print()
    print("  Scoring: 1.0 exact match · 0.5 fascist↔hitler · 0.0 wrong · 'Unknown' excluded")
    print("  Columns:")
    print("    RIA         – liberal-Alice RIA (= Lib(own))")
    print("    N           – belief instances in denominator (liberal games only)")
    print("    Lib(own)    – RIA when Alice was liberal            (= RIA column)")
    print("    Fas(own)    – RIA when Alice was fascist            (sanity check)")
    print("    Hit(own)    – RIA when Alice was Hitler             (sanity check)")
    print("    vs Lib/Fas/Hit – RIA broken down by target's true role (liberal games only)")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ria(
    results: dict[str, tuple[str, RIAStats]],
    out_path: Path,
    win_rates: Optional[dict[str, float]] = None,
) -> None:
    """Grouped bar chart: overall RIA + per-own-role breakdown."""
    models = [
        (folder_key, display_name, stats)
        for folder_key, (display_name, stats) in results.items()
        if stats.total > 0
    ]
    if not models:
        print("No data to plot.")
        return

    # Sort by overall win rate descending (fallback to 0 if unavailable)
    if win_rates is None:
        win_rates = {}
    models.sort(key=lambda x: win_rates.get(x[1], 0), reverse=True)

    n = len(models)
    x = np.arange(n)
    bar_w = 0.6

    lib_own = [s.ria_by_own_role("liberal") or 0 for _, _, s in models]

    display_names = [dn for _, dn, _ in models]
    model_colors  = [plot_config.get_model_color(dn) for dn in display_names]

    fig, ax = plt.subplots(figsize=(plot_config.FIG_WIDTH, 2.5))

    bars = ax.bar(x, lib_own, bar_w, color=model_colors, zorder=4, alpha=0.95)

    # Match aggregate_tokens style: integer labels above bars.
    for bar, val in zip(bars, lib_own):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val * 100:.0f}\\%",
            ha="center",
            va="bottom",
            fontsize=9,
            zorder=6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=35, ha="right")
    ax.tick_params(axis="x", color="0.85", labelcolor="0", pad=0)
    ax.tick_params(axis="y", color="0.85", labelcolor="0")
    ax.set_ylabel(r"Role Identification Accuracy")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}\\%"))
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, n - 0.5)
    ax.margins(x=0, y=0)
    ax.grid(True, alpha=0.3, zorder=0)

    plt.tight_layout(pad=0.01)

    # Render once so tick-label bounding boxes are available
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv_ax = ax.transAxes.inverted()

    for i, display_name in enumerate(display_names):
        imagebox = plot_config.get_model_imagebox(display_name)
        if imagebox is not None:
            label = ax.get_xticklabels()[i]
            bbox = label.get_window_extent(renderer)

            # The label is rotated 35° with ha="right", so the text starts
            # at the lower-left corner of its axis-aligned bounding box.
            # Convert that point to axes-fraction coords (stable across savefig).
            end_x, end_y = inv_ax.transform((bbox.x0, bbox.y0))

            # Rotate the image to match the 35° label angle.
            # Rotate RGB and alpha separately: corners of RGB fill with white
            # (255) while alpha corners fill with 0 (transparent), avoiding
            # the opaque-edge artifact that appears when cval=255 is applied
            # uniformly across all channels.
            img_data = imagebox.get_data()
            if img_data.ndim == 3 and img_data.shape[2] == 4:
                rgb = rotate_image(img_data[..., :3], 35, reshape=True, order=1,
                                   mode='constant', cval=255)
                alpha = rotate_image(img_data[..., 3], 35, reshape=True, order=1,
                                     mode='constant', cval=0)
                rotated_data = np.concatenate([rgb, alpha[..., np.newaxis]], axis=2).astype(img_data.dtype)
            else:
                rotated_data = rotate_image(img_data, 35, reshape=True, order=1,
                                            mode='constant', cval=255)
            rotated_imagebox = OffsetImage(rotated_data, zoom=imagebox.get_zoom())

            # Place the icon at the text-start end of the label
            ab = AnnotationBbox(
                rotated_imagebox,
                xy=(end_x, end_y),
                xycoords="axes fraction",
                frameon=False,
                clip_on=False,
                box_alignment=(0.85, 0.4),
                zorder=1,
            )
            ax.add_artist(ab)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close()
    print(f"Plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Role Identification Accuracy (RIA).")
    parser.add_argument(
        "folder", nargs="?", default=None,
        help="Single eval folder (e.g. runsF2-SONNET46).  "
             "If omitted, all models in MODEL_REGISTRY are analysed.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip chart generation.")
    parser.add_argument(
        "--include-abliterated", action="store_true",
        help="Include abliterated / derestricted model variants.",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run built-in sanity tests and exit.",
    )
    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    if args.folder:
        folders = [EVAL_DIR / args.folder]
    else:
        folders = _discover_model_folders(include_abliterated=args.include_abliterated)

    if not folders:
        print("No model folders found.")
        return

    results: dict[str, tuple[str, RIAStats]] = {}
    win_rates: dict[str, float] = {}
    for folder in folders:
        key = folder.name
        display = plot_config.extract_model_name(key)
        stats = compute_ria_for_folder(folder)
        results[key] = (display, stats)
        wr = plot_config.compute_win_rate(folder)
        if wr is not None:
            win_rates[display] = wr
        n_json = len(list(folder.glob("*_summary.json")))
        ria_val = f"{stats.ria*100:.1f}%" if stats.ria is not None else "N/A"
        print(f"  {display:<34} — {n_json} games  RIA={ria_val}  (N={stats.total})")

    print_ria_table(results, win_rates=win_rates)

    if not args.no_plot:
        out = plot_config.get_plot_path("ria_by_model.pdf")
        plot_ria(results, Path(out), win_rates=win_rates)


# ---------------------------------------------------------------------------
# Built-in tests
# ---------------------------------------------------------------------------

def run_tests() -> None:
    """Run simple sanity checks, print results, exit non-zero on failure."""
    import sys

    passed = 0
    failed = 0

    def check(name: str, got, expected, tol: float = 1e-9):
        nonlocal passed, failed
        if isinstance(expected, float) and got is not None:
            ok = abs(got - expected) < tol
        else:
            ok = (got == expected)
        if ok:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name}  (got {got!r}, expected {expected!r})")

    print("=" * 60)
    print("  RIA built-in tests")
    print("=" * 60)

    # ── _belief_score ─────────────────────────────────────────────
    print("\n_belief_score:")
    check("liberal==liberal",   _belief_score("liberal", "liberal"),   1.0)
    check("fascist==fascist",   _belief_score("fascist", "fascist"),   1.0)
    check("hitler==hitler",     _belief_score("hitler", "hitler"),     1.0)
    check("fascist→hitler",     _belief_score("fascist", "hitler"),    0.5)
    check("hitler→fascist",     _belief_score("hitler", "fascist"),    0.5)
    check("liberal!=fascist",   _belief_score("liberal", "fascist"),   0.0)
    check("fascist!=liberal",   _belief_score("fascist", "liberal"),   0.0)
    check("liberal!=hitler",    _belief_score("liberal", "hitler"),    0.0)
    check("hitler!=liberal",    _belief_score("hitler", "liberal"),    0.0)

    # ── _parse_rapid_assessment ───────────────────────────────────
    print("\n_parse_rapid_assessment:")

    # Simple colon-separated
    t1 = "Bob: Liberal\nCharlie: Fascist\nDavid: Hitler"
    r1 = _parse_rapid_assessment(t1, player_names=["Bob", "Charlie", "David"])
    check("simple Bob",     r1.get("Bob"),     "liberal")
    check("simple Charlie", r1.get("Charlie"), "fascist")
    check("simple David",   r1.get("David"),   "hitler")

    # Unknown is excluded entirely
    t2 = "Bob: Liberal\nCharlie: Unknown\nDavid: Fascist"
    r2 = _parse_rapid_assessment(t2, player_names=["Bob", "Charlie", "David"])
    check("unknown excluded",  "Charlie" not in r2, True)
    check("unknown keeps Bob", r2.get("Bob"), "liberal")

    # Bullet list format
    t3 = "* Bob - Liberal\n* Charlie - Fascist"
    r3 = _parse_rapid_assessment(t3, player_names=["Bob", "Charlie"])
    check("bullet Bob",     r3.get("Bob"),     "liberal")
    check("bullet Charlie", r3.get("Charlie"), "fascist")

    # Only known players are returned
    t4 = "Bob: Liberal\nZara: Fascist"
    r4 = _parse_rapid_assessment(t4, player_names=["Bob"])
    check("only known players", set(r4.keys()), {"Bob"})

    # Empty / None input
    check("empty text",  _parse_rapid_assessment("", player_names=["Bob"]), {})
    check("no players",  _parse_rapid_assessment("Bob: Liberal"), {})

    # ── RIAStats ──────────────────────────────────────────────────
    print("\nRIAStats:")

    s = RIAStats()
    s.add("liberal", "liberal", "liberal")   # 1.0  (exact)
    s.add("fascist", "fascist", "liberal")   # 1.0  (exact)
    s.add("fascist", "hitler",  "liberal")   # 0.5  (half credit)
    s.add("liberal", "fascist", "liberal")   # 0.0  (wrong)
    check("correct=2.5", s.correct, 2.5)
    check("total=4",     s.total,   4)
    check("ria=0.625",   s.ria,     2.5 / 4)

    s2 = RIAStats()
    check("empty ria=None", s2.ria, None)

    # Half credit propagates to by_target_role
    s3 = RIAStats()
    s3.add("hitler", "fascist", "liberal")  # half credit vs fascist target
    check("by_target fascist correct=0.5",
          s3.by_target_role["fascist"]["correct"], 0.5)
    check("by_target fascist total=1",
          s3.by_target_role["fascist"]["total"], 1)

    # ── compute_ria_for_game ──────────────────────────────────────
    print("\ncompute_ria_for_game:")

    game = {
        "players": [
            {"username": "Alice",   "role": "liberal", "_id": "0", "icon": 0},
            {"username": "Bob",     "role": "liberal", "_id": "1", "icon": 1},
            {"username": "Charlie", "role": "fascist", "_id": "2", "icon": 2},
            {"username": "David",   "role": "hitler",  "_id": "3", "icon": 3},
            {"username": "Eve",     "role": "liberal", "_id": "4", "icon": 4},
        ],
        "logs": [
            {
                "rapidAssessments": {
                    "0": "Bob: Liberal\nCharlie: Liberal\nDavid: Fascist\nEve: Liberal",
                }
            }
        ],
    }
    gs = compute_ria_for_game(game)
    a = gs["Alice"]
    # Bob=liberal→liberal  = 1.0
    # Charlie=liberal→fascist = 0.0
    # David=fascist→hitler = 0.5
    # Eve=liberal→liberal  = 1.0
    check("game total=4",     a.total,   4)
    check("game correct=2.5", a.correct, 2.5)
    check("game ria=0.625",   a.ria,     2.5 / 4)

    # Fascist Alice knows teammates → high accuracy
    game2 = {
        "players": [
            {"username": "Alice",   "role": "fascist", "_id": "0", "icon": 0},
            {"username": "Bob",     "role": "liberal", "_id": "1", "icon": 1},
            {"username": "Charlie", "role": "liberal", "_id": "2", "icon": 2},
            {"username": "David",   "role": "hitler",  "_id": "3", "icon": 3},
            {"username": "Eve",     "role": "liberal", "_id": "4", "icon": 4},
        ],
        "logs": [
            {
                "rapidAssessments": {
                    "0": "Bob: Liberal\nCharlie: Liberal\nDavid: Hitler\nEve: Liberal",
                }
            }
        ],
    }
    gs2 = compute_ria_for_game(game2)
    a2 = gs2["Alice"]
    # All correct: 4/4 = 1.0
    check("fascist Alice perfect", a2.ria, 1.0)
    check("fascist own_role",      a2.by_own_role["fascist"]["total"], 4)

    # ── Overall liberal gating ────────────────────────────────────
    print("\nOverall liberal gating (conceptual):")
    # When Alice is fascist, overall RIA should NOT include her stats
    # (only by_own_role["fascist"] captures it as a sanity check)
    combined = RIAStats()
    # simulate: liberal game contributes to overall
    combined.correct += 2.5
    combined.total += 4
    combined.by_own_role["liberal"]["correct"] += 2.5
    combined.by_own_role["liberal"]["total"] += 4
    # simulate: fascist game does NOT contribute to overall
    combined.by_own_role["fascist"]["correct"] += 4.0
    combined.by_own_role["fascist"]["total"] += 4
    check("overall = liberal only",  combined.ria, 2.5 / 4)
    check("fas(own) = sanity 100%",  combined.ria_by_own_role("fascist"), 1.0)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
