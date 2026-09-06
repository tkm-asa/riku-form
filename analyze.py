import os
import cv2
import numpy as np
from ultralytics import YOLO
from enum import Enum

class RunnerType(Enum):
    PITCH = "ピッチ型"
    STRIDE = "ストライド型"
    BALANCED = "バランス型"

class RunningFormAnalyzer:
    def __init__(self, model_path='yolov8l-pose.pt', conf_thresh=0.5):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

        # COCO Pose Keypoint インデックス対応
        self.KP = {
            'nose': 0, 'l_shoulder': 5, 'r_shoulder': 6,
            'l_hip': 11, 'r_hip': 12,
            'l_knee': 13, 'r_knee': 14,
            'l_ankle': 15, 'r_ankle': 16
        }

    def get_main_person_keypoints(self, result):
        """バウンディングボックスの面積が最大の人物のキーポイントと信頼度を取得"""
        if result.boxes is None or len(result.boxes) == 0:
            return None, None
            
        boxes = result.boxes.xywh.cpu().numpy()
        areas = boxes[:, 2] * boxes[:, 3]
        main_idx = np.argmax(areas)
        
        xy = result.keypoints.xy.cpu().numpy()[main_idx].copy()
        conf = result.keypoints.conf.cpu().numpy()[main_idx].copy()
        
        # 信頼度が閾値未満の点は (0, 0) にマスク
        xy[conf < self.conf_thresh] = 0.0
        return xy, conf

    @staticmethod
    def calculate_angle(a, b, c):
        """関節点 b を中心とする点 a, b, c の角度 (度) を計算"""
        try:
            a, b, c = np.array(a), np.array(b), np.array(c)
            if np.all(a == 0) or np.all(b == 0) or np.all(c == 0):
                return None
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return float(np.degrees(angle))
        except Exception:
            return None

    @staticmethod
    def interpolate_keypoints(sequence):
        """
        時系列キーポイント配列の (0, 0) や欠損を線形補間
        sequence: (N_frames, 17, 2)
        """
        n_frames, n_kp, _ = sequence.shape
        interpolated = np.copy(sequence)
        
        for k in range(n_kp):
            for axis in [0, 1]:  # X, Y座標
                vals = interpolated[:, k, axis]
                invalid = (vals == 0) | np.isnan(vals)
                if np.all(invalid):
                    continue
                valid_indices = np.flatnonzero(~invalid)
                invalid_indices = np.flatnonzero(invalid)
                if len(invalid_indices) > 0:
                    vals[invalid] = np.interp(invalid_indices, valid_indices, vals[~invalid])
                interpolated[:, k, axis] = vals
                
        return interpolated

    def estimate_running_direction(self, keypoints_sequence):
        """
        腰のX座標推移からランナーの進行方向を自動推定
        戻り値: 1 (画面左から右へ走行), -1 (画面右から左へ走行)
        """
        hip_x = []
        valid_frames = []
        for i, k in enumerate(keypoints_sequence):
            lh, rh = k[self.KP['l_hip']][0], k[self.KP['r_hip']][0]
            if lh > 0 and rh > 0:
                hip_x.append((lh + rh) / 2.0)
                valid_frames.append(i)
                
        if len(hip_x) < 5:
            return 1  # デフォルトは左→右
            
        slope, _ = np.polyfit(valid_frames, hip_x, 1)
        return 1 if slope >= 0 else -1

    def detect_landing_frames(self, keypoints_sequence, fps=30.0, running_direction=1):
        """
        腰相対座標における足首の水平減速および垂直加速度から接地瞬間（Initial Contact）を検出
        """
        n_frames = len(keypoints_sequence)
        if n_frames < 10:
            return []

        # 欠損補間
        clean_seq = self.interpolate_keypoints(keypoints_sequence)
        
        # 腰中心に対する足首の相対座標系列
        hip_centers = (clean_seq[:, self.KP['l_hip']] + clean_seq[:, self.KP['r_hip']]) / 2.0
        
        l_ankle_rel_x = (clean_seq[:, self.KP['l_ankle'], 0] - hip_centers[:, 0]) * running_direction
        l_ankle_y = clean_seq[:, self.KP['l_ankle'], 1]
        
        r_ankle_rel_x = (clean_seq[:, self.KP['r_ankle'], 0] - hip_centers[:, 0]) * running_direction
        r_ankle_y = clean_seq[:, self.KP['r_ankle'], 1]

        # 速度・加速度の算出
        window = 3
        kernel = np.ones(window) / window
        
        # 平滑化
        l_ax_smooth = np.convolve(l_ankle_rel_x, kernel, mode='same')
        r_ax_smooth = np.convolve(r_ankle_rel_x, kernel, mode='same')
        l_ay_smooth = np.convolve(l_ankle_y, kernel, mode='same')
        r_ay_smooth = np.convolve(r_ankle_y, kernel, mode='same')

        # 水平相対速度 (前方への振り出しが終わり、後方へ引き戻される変曲点を接地とする)
        l_vx = np.gradient(l_ax_smooth)
        r_vx = np.gradient(r_ax_smooth)
        
        # 垂直速度 (下向き速度が止まる瞬間)
        l_vy = np.gradient(l_ay_smooth)
        r_vy = np.gradient(r_ay_smooth)

        landing_candidates = []

        # 相対X座標が極大（足が最も前に出た瞬間）かつ垂直速度が下向きから減速するタイミング
        for i in range(2, n_frames - 2):
            # 左足
            if (l_vx[i-1] >= 0 and l_vx[i+1] < 0) or (l_ax_smooth[i] == np.max(l_ax_smooth[max(0, i-3):min(n_frames, i+4)])):
                if l_ay_smooth[i] > np.mean(l_ay_smooth):  # 下半身にあること
                    landing_candidates.append((i, 'left', l_ax_smooth[i]))
            # 右足
            if (r_vx[i-1] >= 0 and r_vx[i+1] < 0) or (r_ax_smooth[i] == np.max(r_ax_smooth[max(0, i-3):min(n_frames, i+4)])):
                if r_ay_smooth[i] > np.mean(r_ay_smooth):
                    landing_candidates.append((i, 'right', r_ax_smooth[i]))

        landing_candidates.sort(key=lambda x: x[0])
        
        # 近接重複（同じ足の連打や短すぎるインターバル）の除去
        min_interval = int(fps * 0.15)  # 150ms以内の再接地は除外
        filtered_landings = []
        for item in landing_candidates:
            if not filtered_landings:
                filtered_landings.append((item[0], item[1]))
            else:
                prev_frame, prev_side = filtered_landings[-1]
                if item[0] - prev_frame > min_interval:
                    filtered_landings.append((item[0], item[1]))
                    
        return filtered_landings

    def evaluate_form_at_landing(self, keypoints, landing_side, running_direction=1):
        """
        接地瞬間のフォーム指標を多角的に計算
        """
        try:
            ls, rs = keypoints[self.KP['l_shoulder']], keypoints[self.KP['r_shoulder']]
            lh, rh = keypoints[self.KP['l_hip']], keypoints[self.KP['r_hip']]
            
            if landing_side == 'left':
                hip = lh
                knee = keypoints[self.KP['l_knee']]
                ankle = keypoints[self.KP['l_ankle']]
            else:
                hip = rh
                knee = keypoints[self.KP['r_knee']]
                ankle = keypoints[self.KP['r_ankle']]

            if any(np.all(pt == 0) for pt in [ls, rs, lh, rh, knee, ankle]):
                return None

            shoulder_center = (ls + rs) / 2.0
            hip_center = (lh + rh) / 2.0
            
            # 胴体長（スケール正規化用）
            torso_height = np.linalg.norm(shoulder_center - hip_center)
            if torso_height < 1e-3:
                return None

            # 1. 腰の乗り（腰中心に対する接地足首の前後位置）
            # 正: 足が腰より前（オーバーストライド傾向）, 負: 腰が足の上または前方
            forward_offset = (ankle[0] - hip_center[0]) * running_direction / torso_height
            
            # 理想は 0.05 〜 0.20 程度（接地直後に真下に乗れる位置）
            # 0.40 を超えるとブレーキになるオーバーストライド
            if forward_offset <= 0.15:
                hip_load_score = 100.0
            else:
                hip_load_score = max(0.0, 100.0 - (forward_offset - 0.15) * 200.0)

            # 2. 膝の屈曲角 (Knee Angle) - 理想値: 155°〜168°（完全伸展180°の突っ張り着地を防ぐ）
            knee_angle = self.calculate_angle(hip, knee, ankle)
            
            # 3. 体幹前傾角 (Trunk Lean Angle) - 鉛直軸に対する前傾度合い (理想: 5°〜10°)
            torso_vec = shoulder_center - hip_center
            # 画像座標系(y下向き)で鉛直線に対する角度
            trunk_lean = np.degrees(np.arctan2(torso_vec[0] * running_direction, -torso_vec[1]))

            # 4. 脛の傾き角 (Shank Angle at Contact) - 理想はほぼ鉛直〜わずかに前傾 (-5°〜5°)
            shank_vec = knee - ankle
            shank_angle = np.degrees(np.arctan2(shank_vec[0] * running_direction, -shank_vec[1]))

            return {
                'hip_load_score': float(hip_load_score),
                'forward_offset_norm': float(forward_offset),
                'knee_angle': float(knee_angle) if knee_angle else None,
                'trunk_lean': float(trunk_lean),
                'shank_angle': float(shank_angle)
            }
        except Exception:
            return None

    def classify_runner_type(self, landing_count, total_frames, avg_stride_norm, fps=30.0):
        """ピッチ(SPM)とストライドからタイプを分類"""
        duration_sec = total_frames / fps
        if duration_sec <= 0 or landing_count < 2:
            return RunnerType.BALANCED, 0.0

        spm = (landing_count / duration_sec) * 60.0
        
        if spm >= 185 and (avg_stride_norm is None or avg_stride_norm < 1.8):
            rtype = RunnerType.PITCH
        elif spm <= 165 or (avg_stride_norm is not None and avg_stride_norm > 2.2):
            rtype = RunnerType.STRIDE
        else:
            rtype = RunnerType.BALANCED
            
        return rtype, spm

    def analyze_video(self, video_path, output_video_path=None):
        """動画全体の解析パイプラインを実行"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        raw_keypoints = []

        print("1/3: YOLO ポーズ推定を実行中...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame, verbose=False)
            kp, _ = self.get_main_person_keypoints(results[0])
            
            frames.append(frame)
            raw_keypoints.append(kp if kp is not None else np.zeros((17, 2)))

        cap.release()
        keypoints_seq = np.array(raw_keypoints)

        print("2/3: 進行方向推定および接地フレーム検出中...")
        direction = self.estimate_running_direction(keypoints_seq)
        dir_str = "左 -> 右" if direction == 1 else "右 -> 左"
        print(f"  - 推定進行方向: {dir_str}")

        landings = self.detect_landing_frames(keypoints_seq, fps=fps, running_direction=direction)
        print(f"  - 検出接地回数: {len(landings)} 回")

        # フォーム評価の集計
        evaluations = []
        landing_frame_dict = {}
        for f_idx, side in landings:
            eval_res = self.evaluate_form_at_landing(keypoints_seq[f_idx], side, direction)
            if eval_res:
                eval_res['frame'] = f_idx
                eval_res['side'] = side
                evaluations.append(eval_res)
                landing_frame_dict[f_idx] = eval_res

        # ストライド推定（接地間の足首距離を胴体長で正規化）
        strides_norm = []
        for i in range(1, len(landings)):
            f_prev, side_prev = landings[i-1]
            f_curr, side_curr = landings[i]
            if side_prev != side_curr:
                ank_prev = keypoints_seq[f_prev][self.KP[f"{side_prev[0]}_ankle"]]
                ank_curr = keypoints_seq[f_curr][self.KP[f"{side_curr[0]}_ankle"]]
                torso = np.linalg.norm(
                    (keypoints_seq[f_curr][5] + keypoints_seq[f_curr][6])/2 - 
                    (keypoints_seq[f_curr][11] + keypoints_seq[f_curr][12])/2
                )
                if torso > 0:
                    strides_norm.append(abs(ank_curr[0] - ank_prev[0]) / torso)

        avg_stride_norm = np.mean(strides_norm) if strides_norm else None
        runner_type, spm = self.classify_runner_type(len(landings), total_frames, avg_stride_norm, fps)

        print("3/3: 解析結果のサマリー生成中...")
        avg_hip_load = np.mean([e['hip_load_score'] for e in evaluations]) if evaluations else 0.0
        avg_knee_angle = np.mean([e['knee_angle'] for e in evaluations if e['knee_angle']]) if evaluations else 0.0
        avg_trunk_lean = np.mean([e['trunk_lean'] for e in evaluations]) if evaluations else 0.0

        summary = {
            'runner_type': runner_type.value,
            'cadence_spm': round(spm, 1),
            'avg_hip_load_score': round(avg_hip_load, 1),
            'avg_knee_angle': round(avg_knee_angle, 1),
            'avg_trunk_lean': round(avg_trunk_lean, 1),
            'direction': dir_str,
            'total_landings': len(landings)
        }

        # アノテーション動画の出力
        if output_video_path:
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            current_eval = None
            eval_display_timer = 0
            
            for i, frame in enumerate(frames):
                if i in landing_frame_dict:
                    current_eval = landing_frame_dict[i]
                    eval_display_timer = int(fps * 0.4) # 接地情報を0.4秒間表示
                
                # 接地情報のオーバーレイ
                if current_eval and eval_display_timer > 0:
                    eval_display_timer -= 1
                    cv2.putText(frame, f"LANDING ({current_eval['side'].upper()})", (50, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    cv2.putText(frame, f"Hip Load Score: {current_eval['hip_load_score']:.1f}/100", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Knee Angle: {current_eval['knee_angle']:.1f} deg", (50, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, f"Trunk Lean: {current_eval['trunk_lean']:.1f} deg", (50, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # 全体サマリーの常時表示
                cv2.putText(frame, f"SPM: {spm:.1f} | Type: {runner_type.value}", (50, height - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)

                out.write(frame)
            out.release()
            print(f"解析動画を出力しました: {output_video_path}")

        return summary


if __name__ == "__main__":
    # 使用例
    analyzer = RunningFormAnalyzer(model_path='yolov8l-pose.pt')
    
    # 解析対象の動画パス
    input_video = "IMG_1612.mov"
    output_video = None
    
    if os.path.exists(input_video):
        report = analyzer.analyze_video(input_video, output_video_path=output_video)
        print("\n===== ランニングフォーム解析レポート =====")
        for k, v in report.items():
            print(f"{k}: {v}")
    else:
        print(f"ファイル '{input_video}' が見つかりません。パスを確認してください。")