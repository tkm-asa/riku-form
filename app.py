import streamlit as st
import tempfile
import os
import json
from pathlib import Path
import cv2
import numpy as np
from analyze import RunningFormAnalyzer, EvaluationLevel

# ページ設定
st.set_page_config(
    page_title=" ランニングフォーム分析",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# スタイル設定
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.2em;
    }
    </style>
    """, unsafe_allow_html=True)

# ============ サイドバー ============
st.sidebar.title("⚙️ 設定")
st.sidebar.markdown("---")

# YOLOモデル選択
model_option = st.sidebar.selectbox(
    "使用するYOLOモデル",
    ["yolov8l-pose.pt (推奨・精度高)", "yolov8m-pose.pt (軽量)"],
    help="精度と速度のバランスを選択できます"
)

model_path = "yolov8l-pose.pt" if "yolov8l" in model_option else "yolov8m-pose.pt"

# 信頼度閾値
conf_thresh = st.sidebar.slider(
    "キーポイント検出の信頼度閾値",
    min_value=0.3,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="低いほど検出感度が上がりますが、ノイズが増えます"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 使い方")
st.sidebar.markdown("""
1. **動画をアップロード**
   - .mov, .mp4, .avi形式に対応
   - 1分程度の走行動画を推奨

2. **分析を実行**
   - YOLO推定 → 接地検出 → フォーム評価
   - 2-3分程度の時間がかかります

3. **結果を確認**
   - 骨格描画動画
   - フォーム評価とフィードバック
   - 一流選手との比較
""")

# ============ メインコンテンツ ============
st.title("🏃 ランニングフォーム分析システム")
st.markdown("**一流選手の標準値と比較して、あなたのフォームを詳細に分析します**")

st.markdown("---")

# ファイルアップロード
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📹 動画ファイルをアップロード",
        type=["mov", "mp4", "avi"],
        help="スマートフォンで撮影した走行動画をアップロードしてください"
    )

with col2:
    st.markdown("### サンプル動画")
    if st.button("🎥 デモ動画で試す"):
        st.info("デモ動画機能はまだ準備中です")

st.markdown("---")

# 分析実行
if uploaded_file is not None:
    st.success(f"✅ ファイル読み込み完了: {uploaded_file.name}")
    
    # 動画情報表示
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ファイルサイズ", f"{uploaded_file.size / (1024*1024):.1f} MB")
    
    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mov') as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name
    
    # 動画の基本情報を取得
    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps
    cap.release()
    
    with col2:
        st.metric("フレームレート", f"{fps:.0f} fps")
    with col3:
        st.metric("動画長", f"{duration_sec:.1f} 秒")
    
    st.markdown("---")
    
    # 分析ボタン
    if st.button("🚀 フォーム分析を実行", use_container_width=True, type="primary"):
        try:
            # 分析実行
            with st.spinner("🔄 YOLO ポーズ推定を実行中..."):
                analyzer = RunningFormAnalyzer(model_path=model_path, conf_thresh=conf_thresh)
            
            with st.spinner("🔄 接地検出とフォーム評価を実行中..."):
                # 出力動画のパス
                output_video_path = temp_path.replace('.mov', '_analyzed.mp4')
                
                # 分析実行
                report = analyzer.analyze_video(
                    temp_path,
                    output_video_path=output_video_path
                )
            
            st.success("✅ 分析完了！")
            
            # ============ 結果表示 ============
            st.markdown("---")
            st.markdown("## 📊 分析結果")
            
            # セッション情報
            session_info = report['session_info']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("ケイデンス", f"{session_info['cadence_spm']:.1f} SPM")
            with col2:
                st.metric("ランナータイプ", session_info['runner_type'])
            with col3:
                st.metric("接地回数", f"{session_info['total_landings']} 回")
            with col4:
                st.metric("進行方向", session_info['direction'])
            
            st.markdown("---")
            
            # 総合スコア
            overall_score = report['overall_assessment']['avg_score']
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"""
                ### 📈 総合スコア
                
                **{overall_score:.0f} / 100**
                """)
                
                if overall_score >= 85:
                    st.success("🌟 優秀")
                elif overall_score >= 70:
                    st.info("✅ 良好")
                elif overall_score >= 55:
                    st.warning("⚠️ 要改善")
                else:
                    st.error("🔴 要大幅改善")
            
            with col2:
                # スコアゲージ
                progress_color = "green" if overall_score >= 70 else "orange" if overall_score >= 55 else "red"
                st.progress(min(overall_score / 100.0, 1.0))
            
            st.markdown("---")
            
            # タブ表示
            tab1, tab2, tab3, tab4 = st.tabs(["📹 分析動画", "🔍 詳細評価", "💡 改善ポイント", "📋 全接地データ"])
            
            with tab1:
                st.markdown("### 骨格描画付き分析動画")
                st.markdown("*接地時に骨格が強調表示されます*")
                
                if os.path.exists(output_video_path):
                    with open(output_video_path, 'rb') as video_file:
                        st.video(video_file)
                    
                    # ダウンロードボタン
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        with open(output_video_path, 'rb') as f:
                            st.download_button(
                                label="📥 動画をダウンロード",
                                data=f.read(),
                                file_name="running_analysis.mp4",
                                mime="video/mp4",
                                use_container_width=True
                            )
                else:
                    st.warning("⚠️ 分析動画の生成に失敗しました")
            
            with tab2:
                st.markdown("### 各接地フレームの詳細評価")
                
                landing_evals = report['landing_evaluations']
                
                # 接地を選択
                selected_landing = st.selectbox(
                    "評価する接地を選択",
                    range(len(landing_evals)),
                    format_func=lambda x: f"接地#{x+1} ({landing_evals[x]['side']}足 | フレーム{landing_evals[x]['frame']} | スコア{landing_evals[x]['overall_score']:.0f})"
                )
                
                eval_data = landing_evals[selected_landing]
                
                st.markdown(f"#### 接地#{selected_landing+1} - {eval_data['side'].upper()}足")
                st.markdown(f"**総合スコア: {eval_data['overall_score']:.0f} / 100**")
                st.markdown(f"*{eval_data['overall_recommendation']}*")
                
                st.markdown("---")
                
                # フィードバック表示
                for feedback in eval_data['feedback']:
                    priority = feedback['priority']
                    category = feedback['category']
                    level = feedback['level']
                    score = feedback['score']
                    message = feedback['message']
                    advice = feedback['advice']
                    comparison = feedback['comparison']
                    
                    # レベルに応じた表示
                    if level == "優秀":
                        color = "green"
                        icon = "✅"
                    elif level == "良好":
                        color = "blue"
                        icon = "👍"
                    elif level == "要注意":
                        color = "orange"
                        icon = "⚠️"
                    else:  # 改善必須
                        color = "red"
                        icon = "🔴"
                    
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"### {icon} {category}")
                            st.markdown(f"**{message}** ({score:.0f}/100)")
                        with col2:
                            st.markdown(f"**優先度:** {priority}")
                        
                        # 比較情報
                        comp = comparison
                        st.markdown(f"""
                        **あなた:** {comp['your_value']} | **標準値:** {comp['ideal_value']} | **範囲:** {comp['range']}  
                        **差分:** {comp['difference']:+.1f} ({['✅ 範囲内', '❌ 範囲外'][not comp['within_range']]})
                        """)
                        
                        # アドバイス
                        st.info(f"💡 {advice}")
                        st.markdown("---")
            
            with tab3:
                st.markdown("### 優先的に改善すべきポイント")
                
                insights = report.get('key_insights', {})
                
                if insights.get('critical_issue_count', 0) > 0:
                    st.error(f"🔴 改善必須の課題: {insights['critical_issue_count']}項目")
                
                if insights.get('caution_issue_count', 0) > 0:
                    st.warning(f"⚠️ 要注意の課題: {insights['caution_issue_count']}項目")
                
                if insights.get('priority_actions'):
                    st.markdown("#### 改善アクション（優先順）")
                    for i, action in enumerate(insights['priority_actions'], 1):
                        st.markdown(f"{i}. {action}")
                else:
                    st.success("✅ 大きな改善点はありません！")
            
            with tab4:
                st.markdown("### 全接地データ（JSON形式）")
                
                # JSON表示
                with st.expander("📋 詳細JSONデータを表示", expanded=False):
                    st.json(report)
            
            st.markdown("---")
            
            # クリーンアップ
            os.remove(temp_path)
            
        except Exception as e:
            st.error(f"❌ エラーが発生しました: {str(e)}")
            st.error("詳細なエラー情報:")
            st.code(str(e), language="text")
            
            # クリーンアップ
            if os.path.exists(temp_path):
                os.remove(temp_path)

else:
    st.info("📹 左側から動画ファイルをアップロードしてください")
    
    st.markdown("---")
    st.markdown("""
    ## 📌 このシステムについて
    
    ### 特徴
    - **AI駆動**: YOLOv8を使用した高精度のポーズ推定
    - **一流選手基準**: サニブラウン選手などトップアスリートのデータを基準値に採用
    - **詳細フィードバック**: 膝角、体幹前傾、腰の乗りなど5つの指標で複合評価
    - **ビジュアル分析**: 骨格描画で走動作が一目瞭然
    
    ### 分析される項目
    1. **腰の乗り** - 着地時の足と腰の相対位置
    2. **膝の屈曲** - 衝撃吸収の効率
    3. **蹴り出し** - 脛の角度と推進力
    4. **体軸安定性** - 横揺れの有無
    5. **体幹前傾** - 前傾角度の最適性
    
    ### 準備物
    - スマートフォンまたはカメラ
    - 1分程度の走行動画（横向きで撮影推奨）
    - インターネット接続
    """)

st.markdown("---")
st.markdown("### 📞 サポート")
st.markdown("問題が発生した場合は、GitHubのIssuesページでお知らせください。")
