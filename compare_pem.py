"""
detection_pem.json の比較スクリプト
obj_000005 (CAD) と object_seed42_mesh (SAM-3D) の推定結果を並べて表示する
"""

import json
import numpy as np
import argparse
import os


def load_json(path):
    with open(path) as f:
        return json.load(f)


def rotation_error_deg(R1, R2):
    """2つの回転行列の角度誤差 [deg]"""
    R1 = np.array(R1).reshape(3, 3)
    R2 = np.array(R2).reshape(3, 3)
    trace = np.trace(R1 @ R2.T)
    cos_angle = np.clip((trace - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos_angle))


def translation_error_mm(t1, t2):
    return np.linalg.norm(np.array(t1) - np.array(t2))


def match_detections(dets_a, dets_b):
    """bbox の [x,y,w,h] が一致するペアをマッチング"""
    pairs = []
    used_b = set()
    for da in dets_a:
        for ib, db in enumerate(dets_b):
            if ib in used_b:
                continue
            if da["bbox"] == db["bbox"]:
                pairs.append((da, db))
                used_b.add(ib)
                break
        else:
            pairs.append((da, None))
    return pairs


def fmt_r(R):
    R = np.array(R).reshape(3, 3)
    rows = []
    for row in R:
        rows.append("[" + "  ".join(f"{v:+.4f}" for v in row) + "]")
    return "\n        ".join(rows)


def fmt_t(t):
    return f"[{t[0]:+8.1f}, {t[1]:+8.1f}, {t[2]:+8.1f}] mm"


def main():
    parser = argparse.ArgumentParser(description="detection_pem.json 比較")
    parser.add_argument("--a", default=None, help="obj_000005 の detection_pem.json")
    parser.add_argument("--b", default=None, help="seed42   の detection_pem.json")
    parser.add_argument("--label-a", default="obj_000005 (CAD)")
    parser.add_argument("--label-b", default="seed42_mesh (SAM-3D)")
    args = parser.parse_args()

    # デフォルトパス
    base = os.path.dirname(os.path.abspath(__file__))
    path_a = args.a or os.path.join(base, "/home/okada/ws/project/tmp/obj_000005_templates/sam6d_results/detection_pem.json")
    path_b = args.b or os.path.join(base, "/home/okada/ws/project/tmp/server_reconstructions/object_seed42_mesh_templates/sam6d_results/detection_pem.json")

    dets_a = load_json(path_a)
    dets_b = load_json(path_b)

    print(f"=== detection_pem 比較 ===")
    print(f"  A: {path_a}  ({len(dets_a)} 件)")
    print(f"  B: {path_b}  ({len(dets_b)} 件)")
    print()

    pairs = match_detections(dets_a, dets_b)

    # スコアでソート (A基準)
    pairs.sort(key=lambda p: p[0]["score"], reverse=True)

    for idx, (da, db) in enumerate(pairs):
        bbox = da["bbox"]
        score_a = da["score"]
        score_b = db["score"] if db else float("nan")

        print(f"--- 検出 #{idx+1}  bbox={bbox} ---")
        print(f"  score  | {args.label_a}: {score_a:.4f}   {args.label_b}: {score_b:.4f}")

        ta = np.array(da["t"])
        Ra = da["R"]

        if db is not None:
            tb = np.array(db["t"])
            Rb = db["R"]

            t_err = translation_error_mm(ta, tb)
            r_err = rotation_error_deg(Ra, Rb)

            print(f"  t (A)  | {fmt_t(ta)}")
            print(f"  t (B)  | {fmt_t(tb)}")
            print(f"  Δt     | {t_err:.1f} mm")
            print()
            print(f"  R (A)  | {fmt_r(Ra)}")
            print(f"        ")
            print(f"  R (B)  | {fmt_r(Rb)}")
            print(f"  ΔR     | {r_err:.1f} deg")
        else:
            print(f"  t (A)  | {fmt_t(ta)}")
            print(f"  R (A)  | {fmt_r(Ra)}")
            print(f"  B      | (マッチなし)")

        print()

    # サマリ
    print("=== サマリ (bbox マッチ済みペアのみ) ===")
    matched = [(da, db) for da, db in pairs if db is not None]
    if matched:
        t_errs = [translation_error_mm(da["t"], db["t"]) for da, db in matched]
        r_errs = [rotation_error_deg(da["R"], db["R"]) for da, db in matched]
        scores_a = [da["score"] for da, _ in matched]
        scores_b = [db["score"] for _, db in matched]

        # 最高スコア検出のみ
        best_idx = int(np.argmax(scores_a))
        da_best, db_best = matched[best_idx]
        print(f"  最高スコア検出 (score_A={scores_a[best_idx]:.4f}):")
        print(f"    Δt = {t_errs[best_idx]:.1f} mm")
        print(f"    ΔR = {r_errs[best_idx]:.1f} deg")
        print()
        print(f"  全検出 平均:")
        print(f"    Δt = {np.mean(t_errs):.1f} mm  (max {np.max(t_errs):.1f})")
        print(f"    ΔR = {np.mean(r_errs):.1f} deg (max {np.max(r_errs):.1f})")
        print(f"  score A: mean={np.mean(scores_a):.4f}  max={np.max(scores_a):.4f}")
        print(f"  score B: mean={np.mean(scores_b):.4f}  max={np.max(scores_b):.4f}")


if __name__ == "__main__":
    main()
