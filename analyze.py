import os
import cv2
import numpy as np
from ultralytics import YOLO
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json

class RunnerType(Enum):
    PITCH = "ピッチ型"
    STRIDE = "ストライド型"
    BALANCED = "バランス型"

class EvaluationLevel(Enum):
    EXCELLENT = "優秀"
    GOOD = "良好"
    CAUTION = "要注意"
    CRITICAL = "改善必須"


# ============ 一流短距離選手の標準値 ============
ELITE_STANDARDS = {
    'knee_angle': {
        'ideal': 162.5,
        'range': (160, 165),
        'description': '膝角度'
    },
    'composite_hip_load': {
        'ideal': 92,
        'range': (85, 100),
        'description': '複合腰乗りスコア'
    },
    'trunk_lean': {
        'ideal': 7.5,
        'range': (5, 10),
        'description': '体幹前傾角'
    },
    'shank_angle': {
        'ideal': 0,
        'range': (-5, 5),
        'description': '脛の角度'
    },
    'hip_stability': {
        'ideal': 95,
        'range': (85, 100),
        'description': '腰の安定性'
    },
    'push_off_readiness': {
        'ideal': 92,
        'range': (85, 100),
        'description': '蹴り出し準備度'
    },
}

# ============ COCO Pose スケルトン定義 ============
# キーポイント17個を17本の線で接続
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),           # 鼻から目
    (1, 3), (2, 4),           # 目から耳
    (5, 6),                   # 肩から肩
    (5, 7), (7, 9),           # 左腕
    (6, 8), (8, 10),          # 右腕
    (5, 11), (6, 12),         # 肩から腰
    (11, 12),                 # 腰から腰
    (11, 13), (13, 15),       # 左脚
    (12, 14), (14, 16),       # 右脚
]

# キーポイント名（可視化用）
KEYPOINT_NAMES = [
    'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear',
    'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
    'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle'
]


@dataclass
class ComparisonResult:
    """標準値との比較結果"""
    your_value: float
    ideal_value: float
    range_min: float
    range_max: float
    difference: float
    is_within_range: bool


@dataclass
class FeedbackItem:
    """フィードバック項目（標準値比較付き）"""
    priority: int
    category: str
    level: EvaluationLevel
    score: float
    message: str
    comparison: ComparisonResult
    advice: str


@dataclass
class LandingEvaluation:
    """接地時の完全評価"""
    frame: int
    side: str
    overall_score: float
    feedback: List[FeedbackItem]
    overall_recommendation: str


class SkeletonDrawer:
    """骨格描画を担当するクラス"""
    
    @staticmethod
    def draw_skeleton(frame, keypoints, confidence_threshold=0.5, highlight=False):
        """
        キーポイントと骨格を描画
        
        Args:
            frame: 描画対象のフレーム
            keypoints: キーポイント配列 (17, 2)
            confidence_threshold: 信頼度の閾値
            highlight: 接地フレームかどうか（Trueなら色を濃くする）
        
        Returns:
            描画済みのフレーム
        """
        frame_copy = frame.copy()
        
        # 骨格の線を描画
        for connection in SKELETON_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]
            
            # 両端のポイントが有効か確認
            if np.all(start_point != 0) and np.all(end_point != 0):
                start = tuple(map(int, start_point))
                end = tuple(map(int, end_point))
                
                # 接地時は濃い色、非接地時は薄い色
                if highlight:
                    line_color = (0, 255, 0)  # 緑（接地時）
                    thickness = 3
                else:
                    line_color = (100, 200, 100)  # 薄い緑
                    thickness = 2
                
                cv2.line(frame_copy, start, end, line_color, thickness)
        
        # キーポイントを円で描画
        for idx, (x, y) in enumerate(keypoints):
            if x > 0 and y > 0:  # 有効なキーポイント
                point = (int(x), int(y))
                
                # 接地時は大きく、非接地時は小さく
                if highlight:
                    radius = 8
                    color = (0, 255, 0)  # 緑（接地時）
                    thickness = -1  # 塗りつぶし
                else:
                    radius = 5
                    color = (100, 200, 100)  # 薄い緑
                    thickness = -1
                
                cv2.circle(frame_copy, point, radius, color, thickness)
                
                # キーポイント名を小さく表示（接地時のみ）
                if highlight and idx in [11, 12, 13, 14, 15, 16]:  # 下半身のみ表示
                    cv2.putText(frame_copy, KEYPOINT_NAMES[idx], 
                               (point[0] + 10, point[1] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame_copy


class RunningFormAnalyzer:
    def __init__(self, model_path='yolov8l-pose.pt', conf_thresh=0.5):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

        self.KP = {
            'nose': 0, 'l_shoulder': 5, 'r_shoulder': 6,
            'l_hip': 11, 'r_hip': 12,
            'l_knee': 13, 'r_knee': 14,
            'l_ankle': 15, 'r_ankle': 16
        }
        
        self.skeleton_drawer = SkeletonDrawer()

    def get_main_person_keypoints(self, result):
        """バウンディングボックスの面積が最大の人物のキーポイントと信頼度を取得"""
        if result.boxes is None or len(result.boxes) == 0:
            return None, None
            
        boxes = result.boxes.xywh.cpu().numpy()
        areas = boxes[:, 2] * boxes[:, 3]
        main_idx = np.argmax(areas)
        
        xy = result.keypoints.xy.cpu().numpy()[main_idx].copy()
        conf = result.keypoints.conf.cpu().numpy()[main_idx].copy()
        
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
        """時系列キーポイント配列の (0, 0) や欠損を線形補間"""
        n_frames, n_kp, _ = sequence.shape
        interpolated = np.copy(sequence)
        
        for k in range(n_kp):
            for axis in [0, 1]:
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
        """腰のX座標推移からランナーの進行方向を自動推定"""
        hip_x = []
        valid_frames = []
        for i, k in enumerate(keypoints_sequence):
            lh, rh = k[self.KP['l_hip']][0], k[self.KP['r_hip']][0]
            if lh > 0 and rh > 0:
                hip_x.append((lh + rh) / 2.0)
                valid_frames.append(i)
                
        if len(hip_x) < 5:
            return 1
            
        slope, _ = np.polyfit(valid_frames, hip_x, 1)
        return 1 if slope >= 0 else -1

    def detect_landing_frames(self, keypoints_sequence, fps=30.0, running_direction=1):
        """接地瞬間（Initial Contact）を検出"""
        n_frames = len(keypoints_sequence)
        if n_frames < 10:
            return []

        clean_seq = self.interpolate_keypoints(keypoints_sequence)
        
        hip_centers = (clean_seq[:, self.KP['l_hip']] + clean_seq[:, self.KP['r_hip']]) / 2.0
        
        l_ankle_rel_x = (clean_seq[:, self.KP['l_ankle'], 0] - hip_centers[:, 0]) * running_direction
        l_ankle_y = clean_seq[:, self.KP['l_ankle'], 1]
        
        r_ankle_rel_x = (clean_seq[:, self.KP['r_ankle'], 0] - hip_centers[:, 0]) * running_direction
        r_ankle_y = clean_seq[:, self.KP['r_ankle'], 1]

        window = 3
        kernel = np.ones(window) / window
        
        l_ax_smooth = np.convolve(l_ankle_rel_x, kernel, mode='same')
        r_ax_smooth = np.convolve(r_ankle_rel_x, kernel, mode='same')
        l_ay_smooth = np.convolve(l_ankle_y, kernel, mode='same')
        r_ay_smooth = np.convolve(r_ankle_y, kernel, mode='same')

        l_vx = np.gradient(l_ax_smooth)
        r_vx = np.gradient(r_ax_smooth)
        
        l_vy = np.gradient(l_ay_smooth)
        r_vy = np.gradient(r_ay_smooth)

        landing_candidates = []

        for i in range(2, n_frames - 2):
            if (l_vx[i-1] >= 0 and l_vx[i+1] < 0) or (l_ax_smooth[i] == np.max(l_ax_smooth[max(0, i-3):min(n_frames, i+4)])):
                if l_ay_smooth[i] > np.mean(l_ay_smooth):
                    landing_candidates.append((i, 'left', l_ax_smooth[i]))
            if (r_vx[i-1] >= 0 and r_vx[i+1] < 0) or (r_ax_smooth[i] == np.max(r_ax_smooth[max(0, i-3):min(n_frames, i+4)])):
                if r_ay_smooth[i] > np.mean(r_ay_smooth):
                    landing_candidates.append((i, 'right', r_ax_smooth[i]))

        landing_candidates.sort(key=lambda x: x[0])
        
        min_interval = int(fps * 0.15)
        filtered_landings = []
        for item in landing_candidates:
            if not filtered_landings:
                filtered_landings.append((item[0], item[1]))
            else:
                prev_frame, prev_side = filtered_landings[-1]
                if item[0] - prev_frame > min_interval:
                    filtered_landings.append((item[0], item[1]))
                    
        return filtered_landings

    def calculate_landing_metrics(self, keypoints, landing_side, running_direction=1) -> Optional[Dict]:
        """接地時の各種メトリクスを計算（内部用）"""
        try:
            ls, rs = keypoints[self.KP['l_shoulder']], keypoints[self.KP['r_shoulder']]
            lh, rh = keypoints[self.KP['l_hip']], keypoints[self.KP['r_hip']]
            
            if landing_side == 'left':
                knee = keypoints[self.KP['l_knee']]
                ankle = keypoints[self.KP['l_ankle']]
            else:
                knee = keypoints[self.KP['r_knee']]
                ankle = keypoints[self.KP['r_ankle']]

            if any(np.all(pt == 0) for pt in [ls, rs, lh, rh, knee, ankle]):
                return None

            shoulder_center = (ls + rs) / 2.0
            hip_center = (lh + rh) / 2.0
            
            torso_height = np.linalg.norm(shoulder_center - hip_center)
            if torso_height < 1e-3:
                return None

            # 基本測定値
            forward_offset_norm = float((ankle[0] - hip_center[0]) * running_direction / torso_height)
            knee_angle = self.calculate_angle(hip_center, knee, ankle)
            
            torso_vec = shoulder_center - hip_center
            trunk_lean = float(np.degrees(np.arctan2(torso_vec[0] * running_direction, -torso_vec[1])))
            
            shank_vec = knee - ankle
            shank_angle = float(np.degrees(np.arctan2(shank_vec[0] * running_direction, -shank_vec[1])))

            # 拡張指標
            shoulder_width = abs(rs[0] - ls[0])
            hip_width = abs(rh[0] - lh[0])
            hip_stability = 100.0 - abs(shoulder_width - hip_width) / max(shoulder_width, 1) * 50.0
            hip_stability = float(np.clip(hip_stability, 0, 100))
            
            if knee_angle:
                if 155 <= knee_angle <= 168:
                    knee_flex_quality = 100.0
                elif 145 <= knee_angle < 155:
                    knee_flex_quality = 90.0
                elif 168 < knee_angle <= 175:
                    knee_flex_quality = 85.0
                else:
                    knee_flex_quality = max(0.0, 100.0 - abs(knee_angle - 160) * 2.0)
            else:
                knee_flex_quality = 50.0
            
            if shank_angle is not None:
                if abs(shank_angle) <= 10:
                    push_off_readiness = 100.0
                elif abs(shank_angle) <= 20:
                    push_off_readiness = 85.0
                else:
                    push_off_readiness = max(0.0, 100.0 - abs(shank_angle) * 2.0)
            else:
                push_off_readiness = 50.0

            # 複合スコア
            if forward_offset_norm <= 0.15:
                position_score = 100.0
            elif forward_offset_norm <= 0.30:
                position_score = 85.0
            else:
                position_score = max(0.0, 100.0 - (forward_offset_norm - 0.30) * 200.0)
            
            leg_balance_score = (knee_flex_quality + push_off_readiness) / 2.0
            
            composite_hip_load = (position_score * 0.4 + 
                                 hip_stability * 0.3 + 
                                 leg_balance_score * 0.3)
            
            overall_form_score = (position_score + 
                                 knee_flex_quality + 
                                 push_off_readiness + 
                                 hip_stability) / 4.0

            return {
                'forward_offset_norm': forward_offset_norm,
                'knee_angle': knee_angle,
                'trunk_lean': trunk_lean,
                'shank_angle': shank_angle,
                'hip_stability': hip_stability,
                'knee_flex_quality': knee_flex_quality,
                'push_off_readiness': push_off_readiness,
                'composite_hip_load': composite_hip_load,
                'overall_form_score': overall_form_score
            }
        except Exception as e:
            print(f"メトリクス計算エラー: {e}")
            return None

    def compare_with_elite(self, metric_key: str, your_value: float) -> Optional[ComparisonResult]:
        """標準値との比較を実施"""
        if metric_key not in ELITE_STANDARDS:
            return None
        
        standard = ELITE_STANDARDS[metric_key]
        ideal = standard['ideal']
        range_min, range_max = standard['range']
        
        difference = your_value - ideal
        is_within_range = range_min <= your_value <= range_max
        
        return ComparisonResult(
            your_value=round(float(your_value), 1),
            ideal_value=ideal,
            range_min=range_min,
            range_max=range_max,
            difference=round(difference, 1),
            is_within_range=is_within_range
        )

    def generate_feedback(self, metrics: Dict) -> List[FeedbackItem]:
        """メトリクスからフィードバックを生成（標準値比較付き）"""
        feedback_list = []
        
        feedback_list.append(self._evaluate_hip_load(metrics))
        feedback_list.append(self._evaluate_knee_quality(metrics))
        feedback_list.append(self._evaluate_push_off(metrics))
        feedback_list.append(self._evaluate_stability(metrics))
        feedback_list.append(self._evaluate_trunk(metrics))
        
        priority_map = {
            EvaluationLevel.CRITICAL: 1,
            EvaluationLevel.CAUTION: 2,
            EvaluationLevel.GOOD: 3,
            EvaluationLevel.EXCELLENT: 4
        }
        feedback_list.sort(key=lambda x: (priority_map[x.level], -x.score))
        
        for i, item in enumerate(feedback_list, 1):
            item.priority = i
        
        return feedback_list

    def _evaluate_hip_load(self, metrics: Dict) -> FeedbackItem:
        """腰の乗りを評価"""
        score = metrics['composite_hip_load']
        comparison = self.compare_with_elite('composite_hip_load', score)
        offset = metrics['forward_offset_norm']
        
        if score >= 90:
            level = EvaluationLevel.EXCELLENT
            message = "腰の乗りが優秀"
            advice = "現在の接地フォームは非常に良好です。この状態を維持しましょう。"
        elif score >= 75:
            level = EvaluationLevel.GOOD
            message = "腰の乗りは良好"
            if comparison.difference < -5:
                advice = f"標準値より{abs(comparison.difference):.1f}点低いです。足の着地位置をもう少し腰に近づけることで、さらに効率が上がります。"
            else:
                advice = "全体的に良いフォームです。現在のレベルを維持しましょう。"
        elif score >= 55:
            level = EvaluationLevel.CAUTION
            message = "腰の乗りに改善の余地あり"
            if offset > 0.30:
                advice = f"足が腰より{offset:.2f}分の胴体長分前に出ています（標準値より{comparison.difference:.1f}点低い）。着地位置を腰の真下に近づけましょう。"
            else:
                advice = f"標準値より{abs(comparison.difference):.1f}点低い状態です。腰の安定性を高めるため、体幹トレーニングを強化しましょう。"
        else:
            level = EvaluationLevel.CRITICAL
            message = "腰の乗りが悪化"
            if offset > 0.40:
                advice = f"著しくオーバーストライドしています（標準値より{abs(comparison.difference):.1f}点低い）。着地点を大幅に修正し、ピッチを上げることが急務です。"
            else:
                advice = f"腰が不安定で、標準値より{abs(comparison.difference):.1f}点低い状態です。体幹トレーニングと走動作の見直しが必須です。"
        
        return FeedbackItem(
            priority=0,
            category="腰の乗り",
            level=level,
            score=score,
            message=message,
            comparison=comparison,
            advice=advice
        )

    def _evaluate_knee_quality(self, metrics: Dict) -> FeedbackItem:
        """膝の屈曲品質を評価"""
        score = metrics['knee_flex_quality']
        angle = metrics['knee_angle']
        comparison = self.compare_with_elite('knee_angle', angle) if angle else None
        
        if score >= 95:
            level = EvaluationLevel.EXCELLENT
            message = "膝の屈曲が理想的"
            advice = f"膝角が{angle:.1f}°で標準値{comparison.range_min}-{comparison.range_max}°の中心付近です。着地時の衝撃吸収が最適です。"
        elif score >= 80:
            level = EvaluationLevel.GOOD
            message = "膝の屈曲は良好"
            advice = f"膝角が{angle:.1f}°で標準値内です。十分な衝撃吸収ができています。"
        elif score >= 60:
            level = EvaluationLevel.CAUTION
            message = "膝が硬い、または柔らかすぎる"
            if angle and angle < 155:
                advice = f"膝角が{angle:.1f}°で硬すぎます（標準値160-165°より{abs(angle - 162.5):.1f}°低い）。着地時により深く膝を曲げてください。"
            else:
                advice = f"膝角が{angle:.1f}°で曲がりすぎています（標準値より{abs(angle - 162.5):.1f}°高い）。よりコンパクトな脚動作を心がけてください。"
        else:
            level = EvaluationLevel.CRITICAL
            message = "膝の使い方に大きな課題"
            advice = f"膝角が{angle:.1f}°と大きく標準値から外れています。膝の屈伸動作が不効率です。ドリルトレーニングやコーチの指導を受けることを強く推奨します。"
        
        return FeedbackItem(
            priority=0,
            category="膝の屈曲",
            level=level,
            score=score,
            message=message,
            comparison=comparison,
            advice=advice
        )

    def _evaluate_push_off(self, metrics: Dict) -> FeedbackItem:
        """蹴り出し準備度を評価"""
        score = metrics['push_off_readiness']
        shank = metrics['shank_angle']
        comparison = self.compare_with_elite('shank_angle', shank)
        
        if score >= 95:
            level = EvaluationLevel.EXCELLENT
            message = "蹴り出しの準備が最適"
            advice = f"脛角が{shank:.1f}°でほぼ鉛直です。効率的な蹴り出しができています。"
        elif score >= 80:
            level = EvaluationLevel.GOOD
            message = "蹴り出しの準備は良好"
            advice = f"脛角が{shank:.1f}°で良好な状態です。適切な蹴り出しができています。"
        elif score >= 55:
            level = EvaluationLevel.CAUTION
            message = "蹴り出し動作に改善の余地"
            advice = f"脛角が{shank:.1f}°で、標準値（{comparison.range_min}-{comparison.range_max}°）から{abs(comparison.difference):.1f}°外れています。脛を鉛直に近づけましょう。"
        else:
            level = EvaluationLevel.CRITICAL
            message = "蹴り出し効率が低い"
            advice = f"脛角が{shank:.1f}°と標準値から大きく外れています。蹴り出し効率を大幅に改善できるポイントです。"
        
        return FeedbackItem(
            priority=0,
            category="蹴り出し",
            level=level,
            score=score,
            message=message,
            comparison=comparison,
            advice=advice
        )

    def _evaluate_stability(self, metrics: Dict) -> FeedbackItem:
        """体軸安定性を評価"""
        score = metrics['hip_stability']
        comparison = self.compare_with_elite('hip_stability', score)
        
        if score >= 90:
            level = EvaluationLevel.EXCELLENT
            message = "体軸が安定している"
            advice = f"腰の安定性が{score:.1f}点で標準値に近い状態です。横ぶれのない安定した走りができています。"
        elif score >= 75:
            level = EvaluationLevel.GOOD
            message = "体軸は比較的安定"
            advice = f"腰の安定性が{score:.1f}点で良好です。特に問題ありません。"
        elif score >= 55:
            level = EvaluationLevel.CAUTION
            message = "体軸が不安定な傾向"
            advice = f"腰の安定性が{score:.1f}点で、標準値より{abs(comparison.difference):.1f}点低いです。体幹の安定性を高めるトレーニングを意識しましょう。"
        else:
            level = EvaluationLevel.CRITICAL
            message = "体軸が大きく不安定"
            advice = f"腰の安定性が{score:.1f}点と非常に低い状態です。体幹トレーニングと走動作全体の見直しが急務です。"
        
        return FeedbackItem(
            priority=0,
            category="体軸安定性",
            level=level,
            score=score,
            message=message,
            comparison=comparison,
            advice=advice
        )

    def _evaluate_trunk(self, metrics: Dict) -> FeedbackItem:
        """体幹前傾角を評価"""
        angle = metrics['trunk_lean']
        comparison = self.compare_with_elite('trunk_lean', angle)
        
        if 5 <= angle <= 10:
            level = EvaluationLevel.EXCELLENT
            score = 100.0
            message = "体幹前傾が理想的"
            advice = f"前傾角が{angle:.1f}°で標準値{comparison.range_min}-{comparison.range_max}°の理想的な範囲です。このフォームを維持しましょう。"
        elif 3 <= angle <= 12:
            level = EvaluationLevel.GOOD
            score = 85.0
            message = "体幹前傾は良好"
            advice = f"前傾角が{angle:.1f}°で良好な状態です。大きな問題ありません。"
        elif 0 <= angle < 3 or 12 < angle <= 15:
            level = EvaluationLevel.CAUTION
            score = 65.0
            message = "体幹の角度を調整"
            if angle < 3:
                advice = f"前傾角が{angle:.1f}°で標準値より{abs(comparison.difference):.1f}°不足しています。もう少し前に傾いてください。"
            else:
                advice = f"前傾角が{angle:.1f}°で標準値より{comparison.difference:.1f}°大きいです。腰を立てる意識を持ってください。"
        else:
            level = EvaluationLevel.CRITICAL
            score = 40.0
            message = "体幹前傾に大きな課題"
            if angle < 0:
                advice = f"体幹が{angle:.1f}°と後傾しています。前傾姿勢を意識してください。"
            else:
                advice = f"前傾角が{angle:.1f}°と大きすぎます（標準値より{comparison.difference:.1f}°）。腰をもっと立ててください。"
        
        return FeedbackItem(
            priority=0,
            category="体幹前傾",
            level=level,
            score=score,
            message=message,
            comparison=comparison,
            advice=advice
        )

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

        print("1/4: YOLO ポーズ推定を実行中...")
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

        print("2/4: 進行方向推定および接地フレーム検出中...")
        direction = self.estimate_running_direction(keypoints_seq)
        dir_str = "左 → 右" if direction == 1 else "右 → 左"
        print(f"  - 推定進行方向: {dir_str}")

        landings = self.detect_landing_frames(keypoints_seq, fps=fps, running_direction=direction)
        print(f"  - 検出接地回数: {len(landings)} 回")

        print("3/4: 詳細なフォーム評価を実行中...")
        evaluations: List[LandingEvaluation] = []
        landing_frame_dict = {}
        
        for f_idx, side in landings:
            metrics = self.calculate_landing_metrics(keypoints_seq[f_idx], side, direction)
            if metrics:
                feedback = self.generate_feedback(metrics)
                
                # 総合推奨
                critical_count = sum(1 for f in feedback if f.level == EvaluationLevel.CRITICAL)
                caution_count = sum(1 for f in feedback if f.level == EvaluationLevel.CAUTION)
                
                if critical_count > 0:
                    overall_rec = "⚠️ フォームの大幅な改善が必要です"
                elif caution_count > 2:
                    overall_rec = "📊 複数の項目で改善が必要です"
                elif caution_count > 0:
                    overall_rec = "💡 いくつかの項目を改善することで、さらに効率的な走りになります"
                else:
                    overall_rec = "✅ フォームは良好です。現在のレベルを維持しましょう"
                
                overall_score = metrics['overall_form_score']
                
                evaluation = LandingEvaluation(
                    frame=f_idx,
                    side=side,
                    overall_score=overall_score,
                    feedback=feedback,
                    overall_recommendation=overall_rec
                )
                evaluations.append(evaluation)
                landing_frame_dict[f_idx] = evaluation

        # ストライド計算
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

        print("4/4: レポートを生成中...")
        
        avg_overall = np.mean([e.overall_score for e in evaluations]) if evaluations else 0.0
        
        duration_sec = total_frames / fps

        # JSONシリアライズ用の変換
        def serialize_evaluation(e: LandingEvaluation) -> Dict:
            return {
                'frame': e.frame,
                'side': e.side,
                'overall_score': round(float(e.overall_score), 1),
                'feedback': [
                    {
                        'priority': f.priority,
                        'category': f.category,
                        'level': f.level.value,
                        'score': round(float(f.score), 1),
                        'message': f.message,
                        'comparison': {
                            'your_value': f.comparison.your_value,
                            'ideal_value': f.comparison.ideal_value,
                            'range': f"{f.comparison.range_min}-{f.comparison.range_max}",
                            'difference': f.comparison.difference,
                            'within_range': f.comparison.is_within_range,
                        },
                        'advice': f.advice,
                    }
                    for f in e.feedback
                ],
                'overall_recommendation': e.overall_recommendation,
            }

        summary = {
            'session_info': {
                'direction': dir_str,
                'total_landings': len(landings),
                'duration_sec': round(duration_sec, 2),
                'cadence_spm': round(spm, 1),
                'runner_type': runner_type.value,
            },
            'overall_assessment': {
                'avg_score': round(avg_overall, 1),
                'summary': '一流選手との比較において、あなたのフォーム改善ポイントが以下に示されています。'
            },
            'landing_evaluations': [serialize_evaluation(e) for e in evaluations],
        }

        # キーインサイト生成
        if evaluations:
            all_feedback = []
            for e in evaluations:
                all_feedback.extend(e.feedback)
            
            critical_items = [f for f in all_feedback if f.level == EvaluationLevel.CRITICAL]
            caution_items = [f for f in all_feedback if f.level == EvaluationLevel.CAUTION]
            
            insights = {
                'critical_issue_count': len(critical_items),
                'caution_issue_count': len(caution_items),
                'priority_actions': [],
            }
            
            if critical_items:
                issue_counts = {}
                for item in critical_items:
                    issue_counts[item.category] = issue_counts.get(item.category, 0) + 1
                most_common = max(issue_counts.items(), key=lambda x: x[1])[0]
                insights['priority_actions'].append(f"【最優先】{most_common}の改善に取り組んでください")
            
            if caution_items and len(insights['priority_actions']) < 3:
                unique_categories = set()
                for item in caution_items:
                    if item.category not in unique_categories:
                        insights['priority_actions'].append(f"【次点】{item.message}：{item.advice[:50]}...")
                        unique_categories.add(item.category)
                        if len(insights['priority_actions']) >= 3:
                            break
            
            summary['key_insights'] = insights

        # アノテーション動画出力
        if output_video_path:
            self._write_annotated_video(frames, keypoints_seq, landing_frame_dict, fps, width, height, output_video_path, spm, runner_type)

        return summary

    def _write_annotated_video(self, frames, keypoints_seq, landing_frame_dict, fps, width, height, output_path, spm, runner_type):
        """骨格描画とフィードバック情報をアノテーションした動画を出力"""
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        current_eval = None
        eval_display_timer = 0
        
        for i, frame in enumerate(frames):
            # 全フレームに骨格を描画（接地時はハイライト）
            is_landing = i in landing_frame_dict
            frame_with_skeleton = self.skeleton_drawer.draw_skeleton(
                frame, 
                keypoints_seq[i], 
                highlight=is_landing
            )
            
            if i in landing_frame_dict:
                current_eval = landing_frame_dict[i]
                eval_display_timer = int(fps * 0.5)
            
            # 接地時のフィードバック表示
            if current_eval and eval_display_timer > 0:
                eval_display_timer -= 1
                
                y_pos = 60
                # 背景の半透明矩形
                cv2.rectangle(frame_with_skeleton, (40, 50), (600, 300), (0, 0, 0), -1)
                cv2.rectangle(frame_with_skeleton, (40, 50), (600, 300), (255, 255, 255), 2)
                
                cv2.putText(frame_with_skeleton, f"LANDING ({current_eval.side.upper()}) - Score: {current_eval.overall_score:.0f}/100",
                            (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                
                y_pos += 50
                # 最優先の3つのフィードバックを表示
                for fb in sorted(current_eval.feedback, key=lambda x: x.priority)[:3]:
                    color = (0, 255, 0) if fb.level == EvaluationLevel.EXCELLENT else \
                            (0, 255, 255) if fb.level == EvaluationLevel.GOOD else \
                            (0, 165, 255) if fb.level == EvaluationLevel.CAUTION else (0, 0, 255)
                    
                    comp = fb.comparison
                    diff_str = f"+{comp.difference:.1f}" if comp.difference > 0 else f"{comp.difference:.1f}"
                    cv2.putText(frame_with_skeleton, f"{fb.category}: {fb.message} (標準{diff_str})",
                                (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                    y_pos += 35

            # 全体サマリーの常時表示
            cv2.putText(frame_with_skeleton, f"SPM: {spm:.1f} | Type: {runner_type.value}", (50, height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)

            out.write(frame_with_skeleton)
        out.release()
        print(f"✅ 骨格描画付き解析動画を出力しました: {output_path}")


if __name__ == "__main__":
    analyzer = RunningFormAnalyzer(model_path='yolov8l-pose.pt')
    
    input_video = "IMG_1612.mov"
    output_video = "running_analysis_with_skeleton.mp4"
    
    if os.path.exists(input_video):
        report = analyzer.analyze_video(input_video, output_video_path=output_video)
        
        print("\n" + "="*80)
        print("ランニングフォーム解析レポート（骨格描画版）")
        print("="*80)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        
    else:
        print(f"❌ ファイル '{input_video}' が見つかりません。パスを確認してください。")