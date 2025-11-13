import streamlit as st
import pandas as pd
import numpy as np
import math
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="标准化高斯2SFCA分析工具",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class NormalizedGaussian2SFCA:
    """标准化高斯两步移动搜索法分析类"""
    
    def __init__(self, l0_distance, cost_type="距离"):
        self.l0_distance = l0_distance
        self.cost_type = cost_type
        self._init_gaussian_constants()
        
    def _init_gaussian_constants(self):
        """初始化用户公式的常数项"""
        if self.l0_distance <= 0:
            self._boundary_const = 0.0
            self._denominator = 1.0
            return
            
        self._boundary_const = math.exp(-0.5)
        self._denominator = 1 - self._boundary_const
        
    def gaussian_weight(self, distance):
        """标准化高斯权重函数"""
        if distance >= self.l0_distance:
            return 0.0
        if self._denominator <= 0:
            return 0.0
            
        ratio_squared = (distance / self.l0_distance) ** 2
        weight_unnormalized = math.exp(-0.5 * ratio_squared)
        numerator = weight_unnormalized - self._boundary_const
        weight = numerator / self._denominator
        
        return max(0.0, weight)
    
    def calculate_accessibility(self, df):
        """计算可达性得分"""
        df = df.copy()
        df['UserFormulaWeight'] = df['TravelCost'].apply(self.gaussian_weight)
        
        # 计算每个供给点的加权需求
        supply_demand = defaultdict(float)
        for _, row in df.iterrows():
            if row['UserFormulaWeight'] > 0:
                supply_demand[row['SupplyID']] += row['Demand'] * row['UserFormulaWeight']
        
        # 计算供给比率
        supply_ratios = {}
        supply_data = df[['SupplyID', 'Supply']].drop_duplicates()
        for _, row in supply_data.iterrows():
            supply_id = row['SupplyID']
            supply_value = row['Supply']
            weighted_demand = supply_demand.get(supply_id, 0)
            supply_ratios[supply_id] = supply_value / weighted_demand if weighted_demand > 0 else 0
        
        # 计算每个需求点的可达性
        accessibility_scores = defaultdict(float)
        for _, row in df.iterrows():
            if row['UserFormulaWeight'] > 0:
                origin_id = row['DemandID']
                dest_id = row['SupplyID']
                weight = row['UserFormulaWeight']
                if dest_id in supply_ratios:
                    accessibility_scores[origin_id] += supply_ratios[dest_id] * weight
        
        # 创建结果DataFrame
        demand_points = df[['DemandID', 'Demand']].drop_duplicates()
        results = []
        for _, row in demand_points.iterrows():
            demand_id = row['DemandID']
            results.append({
                'DemandID': demand_id,
                'Demand': row['Demand'],
                'AccessibilityScore': accessibility_scores.get(demand_id, 0)
            })
        
        return pd.DataFrame(results), df, supply_ratios

def create_sample_data():
    """创建示例数据"""
    sample_data = {
        'DemandID': [3, 59, 29, 80, 131, 39, 132, 197, 99],
        'Demand': [35023, 36080, 24316, 26139, 41445, 34871, 24155, 28886, 45856],
        'SupplyID': [215, 215, 215, 215, 215, 215, 215, 215, 215],
        'Supply': [437, 437, 437, 437, 437, 437, 437, 437, 437],
        'TravelCost': [0.05, 8.95, 9.27, 10, 10.98, 12.77, 13.23, 13.68, 14.27]
    }
    return pd.DataFrame(sample_data)

def plot_gaussian_decay(l0_distance, cost_type):
    """绘制高斯衰减曲线"""
    distances = np.linspace(0, l0_distance * 1.5, 100)
    analyzer = NormalizedGaussian2SFCA(l0_distance, cost_type)
    weights = [analyzer.gaussian_weight(d) for d in distances]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=distances, y=weights, mode='lines', name='标准化高斯权重',
        line=dict(color='#FF4B4B', width=3)
    ))
    
    fig.add_vline(x=l0_distance, line_dash="dash", line_color="blue", 
                  annotation_text=f"截止距离 l_0 = {l0_distance}")
    
    fig.update_layout(
        title=f'标准化高斯衰减函数 (l_0 = {l0_distance})',
        xaxis_title=f'{cost_type}',
        yaxis_title='权重',
        showlegend=True,
        height=400,
        template="plotly_white"
    )
    
    return fig

def plot_accessibility_distribution(results_df):
    """绘制可达性分布"""
    fig = px.histogram(
        results_df, x='AccessibilityScore', title='可达性得分分布',
        nbins=20, color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(
        xaxis_title='可达性得分', yaxis_title='频数', height=400,
        template="plotly_white"
    )
    return fig

def plot_od_connections(df_with_weights, cost_type):
    """绘制OD连接权重分布"""
    fig = px.scatter(
        df_with_weights, x='TravelCost', y='UserFormulaWeight',
        size='Demand', color='UserFormulaWeight', title='OD连接权重分布',
        hover_data=['DemandID', 'SupplyID'], color_continuous_scale='viridis'
    )
    fig.update_layout(
        xaxis_title=f'{cost_type}成本', yaxis_title='用户公式权重', 
        height=400, template="plotly_white"
    )
    return fig

def main():
    """主应用函数"""
    
    # 应用标题和介绍
    st.markdown('<h1 class="main-header">🏥 标准化高斯2SFCA可达性分析工具</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="info-box">
        <b>📊 工具介绍：</b> 基于标准化高斯两步移动搜索法的空间可达性分析工具，
        用于评估服务设施（医院、学校等）的空间可达性分布。
        </div>
        """, unsafe_allow_html=True)
    
    # 侧边栏 - 参数配置
    with st.sidebar:
        st.header("⚙️ 分析配置")
        
        # 成本类型选择
        cost_type = st.selectbox(
            "出行成本类型",
            ["距离", "时间"],
            help="选择TravelCost列的单位类型"
        )
        
        cost_unit = st.text_input(
            "成本单位",
            value="米" if cost_type == "距离" else "分钟",
            help="例如：米、公里、分钟、小时等"
        )
        
        st.markdown("---")
        st.subheader("📏 截止距离参数")
        
        st.markdown("""
        **l₀ 参数说明：**
        - 空间相互作用的最大范围
        - 超过此值的权重为0
        - 影响衰减曲线形状
        """)
        
        # 根据成本类型设置不同的默认值和范围
        if cost_type == "距离":
            default_l0, min_val, max_val = 15.0, 0.1, 10000.0
            presets = {"步行尺度 (800米)": 800, "自行车尺度 (3000米)": 3000, "驾车尺度 (10000米)": 10000}
        else:
            default_l0, min_val, max_val = 30.0, 0.1, 600.0
            presets = {"步行尺度 (15分钟)": 15, "自行车尺度 (30分钟)": 30, "驾车尺度 (60分钟)": 60}
        
        # 预设按钮
        selected_preset = st.selectbox("快速设置", ["自定义"] + list(presets.keys()))
        if selected_preset != "自定义":
            l0_distance = presets[selected_preset]
            st.info(f"已选择: {selected_preset}")
        else:
            l0_distance = st.slider(
                f"截止距离 l₀ ({cost_unit})",
                min_value=min_val, max_value=max_val, value=default_l0, step=1.0
            )
        
        st.markdown("---")
        st.subheader("📁 数据输入")
        
        uploaded_file = st.file_uploader(
            "上传CSV文件",
            type=['csv'],
            help="文件应包含: DemandID, Demand, SupplyID, Supply, TravelCost"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                required_columns = ['DemandID', 'Demand', 'SupplyID', 'Supply', 'TravelCost']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ 缺少列: {missing_columns}")
                    st.info("使用示例数据")
                    df = create_sample_data()
                else:
                    st.success("✅ 数据上传成功!")
            except Exception as e:
                st.error(f"❌ 文件读取错误: {str(e)}")
                st.info("使用示例数据")
                df = create_sample_data()
        else:
            st.info("使用示例数据")
            df = create_sample_data()
    
    # 主内容区 - 数据展示
    st.markdown('<div class="section-header">📋 数据概览</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(df, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>📊 数据统计</h3>
        """, unsafe_allow_html=True)
        st.write(f"**需求点数量:** {df['DemandID'].nunique()}")
        st.write(f"**供给点数量:** {df['SupplyID'].nunique()}")
        st.write(f"**OD连接数量:** {len(df)}")
        st.write(f"**平均{cost_type}成本:** {df['TravelCost'].mean():.2f} {cost_unit}")
        st.write(f"**最大{cost_type}成本:** {df['TravelCost'].max():.2f} {cost_unit}")
        st.write(f"**最小{cost_type}成本:** {df['TravelCost'].min():.2f} {cost_unit}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
        <h3>⚙️ 分析参数</h3>
        """, unsafe_allow_html=True)
        st.write(f"**成本类型:** {cost_type}")
        st.write(f"**截止距离 l₀:** {l0_distance} {cost_unit}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # 分析按钮
    st.markdown("---")
    if st.button("🚀 开始可达性分析", type="primary", use_container_width=True):
        with st.spinner("正在进行可达性分析..."):
            try:
                # 执行分析
                analyzer = NormalizedGaussian2SFCA(l0_distance, cost_type)
                results_df, df_with_weights, supply_ratios = analyzer.calculate_accessibility(df)
                
                # 显示成功信息
                st.markdown("""
                <div class="success-box">
                <h3>✅ 分析完成！</h3>
                可达性分析已成功完成，以下是详细结果。
                </div>
                """, unsafe_allow_html=True)
                
                # 结果显示
                st.markdown('<div class="section-header">📈 分析结果</div>', unsafe_allow_html=True)
                
                # 统计摘要
                col1, col2, col3, col4 = st.columns(4)
                accessibility_scores = results_df['AccessibilityScore']
                
                with col1:
                    st.metric("平均可达性", f"{accessibility_scores.mean():.6f}")
                with col2:
                    st.metric("最大可达性", f"{accessibility_scores.max():.6f}")
                with col3:
                    st.metric("最小可达性", f"{accessibility_scores.min():.6f}")
                with col4:
                    zero_count = (accessibility_scores == 0).sum()
                    st.metric("零可达性点", f"{zero_count}/{len(results_df)}")
                
                # 结果表格
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🏆 可达性排名")
                    display_df = results_df.sort_values('AccessibilityScore', ascending=False)
                    display_df['排名'] = range(1, len(display_df) + 1)
                    st.dataframe(display_df[['排名', 'DemandID', 'Demand', 'AccessibilityScore']], 
                               use_container_width=True)
                
                with col2:
                    st.subheader("⚖️ 供给比率")
                    supply_df = pd.DataFrame([
                        {'供给点ID': k, '供给比率': v} 
                        for k, v in supply_ratios.items()
                    ])
                    st.dataframe(supply_df, use_container_width=True)
                
                # 可视化分析
                st.markdown('<div class="section-header">📊 可视化分析</div>', unsafe_allow_html=True)
                
                # 第一行图表
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_decay = plot_gaussian_decay(l0_distance, cost_type)
                    st.plotly_chart(fig_decay, use_container_width=True)
                
                with col2:
                    fig_dist = plot_accessibility_distribution(results_df)
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                # 第二行图表
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_od = plot_od_connections(df_with_weights, cost_type)
                    st.plotly_chart(fig_od, use_container_width=True)
                
                with col2:
                    top_results = results_df.nlargest(min(10, len(results_df)), 'AccessibilityScore')
                    fig_rank = px.bar(
                        top_results, x='DemandID', y='AccessibilityScore',
                        title='🏅 Top 10 可达性得分排名', color='AccessibilityScore',
                        color_continuous_scale='viridis'
                    )
                    fig_rank.update_layout(height=400, template="plotly_white")
                    st.plotly_chart(fig_rank, use_container_width=True)
                
                # 技术细节
                with st.expander("🔬 技术细节", expanded=False):
                    st.subheader("公式常数")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**边界常数 e^(-1/2):** {analyzer._boundary_const:.6f}")
                    with col2:
                        st.write(f"**标准化分母:** {analyzer._denominator:.6f}")
                    
                    st.subheader("权重计算示例")
                    test_data = []
                    test_distances = [0, l0_distance*0.25, l0_distance*0.5, l0_distance*0.75, l0_distance]
                    for dist in test_distances:
                        weight = analyzer.gaussian_weight(dist)
                        ratio = dist / l0_distance if l0_distance > 0 else 0
                        test_data.append({
                            f'{cost_type}({cost_unit})': f"{dist:.2f}",
                            'l_rn/l_0': f"{ratio:.2f}",
                            '权重': f"{weight:.4f}"
                        })
                    st.table(pd.DataFrame(test_data))
                
                # 下载结果
                st.markdown('<div class="section-header">💾 下载结果</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_results = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 下载可达性结果 (CSV)",
                        data=csv_results,
                        file_name=f"可达性分析结果_l0_{l0_distance}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    csv_od = df_with_weights.to_csv(index=False)
                    st.download_button(
                        label="📥 下载详细OD数据 (CSV)",
                        data=csv_od,
                        file_name=f"OD连接数据_l0_{l0_distance}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"❌ 分析过程中出现错误: {str(e)}")
                st.info("请检查数据格式和参数设置，或联系技术支持")
    
    # 使用说明
    with st.expander("📖 使用指南", expanded=False):
        st.markdown("""
        ### 🎯 使用步骤
        
        1. **准备数据**：确保CSV文件包含以下列：
           - `DemandID` - 需求点唯一标识
           - `Demand` - 需求量（如人口数量）
           - `SupplyID` - 供给点唯一标识  
           - `Supply` - 供给量（如服务设施容量）
           - `TravelCost` - 出行成本（距离或时间）
        
        2. **设置参数**：
           - 选择出行成本类型（距离或时间）
           - 设置合适的截止距离 l₀
           - 上传数据文件或使用示例数据
        
        3. **运行分析**：点击"开始可达性分析"按钮
        
        4. **查看结果**：分析结果包括：
           - 可达性得分表格和排名
           - 统计摘要
           - 可视化图表
           - 可下载的结果文件
        
        ### 📏 参数设定建议
        
        **截止距离 l₀ 设定：**
        - **距离类型**：步行800-1500米，驾车5000-15000米
        - **时间类型**：步行15-30分钟，驾车30-60分钟
        
        ### 📊 结果解释
        
        - **可达性得分**：数值越高表示可达性越好
        - **供给比率**：供给量与加权需求的比值
        - **权重衰减**：反映空间相互作用的衰减模式
        
        ### ❓ 常见问题
        
        **Q: 为什么有些点的可达性得分为0？**
        A: 这可能是因为该需求点与所有供给点的距离都超过了截止距离 l₀
        
        **Q: 如何选择合适的 l₀ 值？**
        A: 根据实际出行行为和研究目的选择，可参考预设值或进行敏感性分析
        """)

if __name__ == "__main__":
    main()
