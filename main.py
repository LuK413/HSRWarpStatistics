import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from statsmodels.distributions.empirical_distribution import ECDF

# https://www.reddit.com/r/Genshin_Impact/comments/jo9d9d/the_5_rate_is_not_uniform_06_there_is_a_soft_pity/
# https://www.prydwen.gg/star-rail/guides/gacha-system/

st.set_page_config(page_title='HSR Warp Calculator', page_icon='stellar_jade.jpg')

def limited_wish(initial_pity, banner_type, last_five_star, num_limited):
    """
    initial_pity: Integer from 0 to 89
    banner_type: Can be only 'Character' or 'Light Cone'
    last_five_star: 'Limited' or 'Standard' depending on what the last pulled 5 star was
    num_limited: The targeted number of limited 5 stars to pull
    """
    wishes = []
    limited = 0
    if banner_type == 'Character':
        soft_pity_thresh = 73
        guarantee = 90
        five_star_rate = 0.006
    else:
        soft_pity_thresh = 62
        guarantee = 80
        five_star_rate = 0.008
    five_star_map = {0: 'Not 5 Star', 1: '5 Star'}    
    fiftyfifty_map = {0: 'Standard', 1: 'Limited'}
    prob_increase = (1 - five_star_rate) / (guarantee - soft_pity_thresh)
    initial_pity_flag = True
    while limited < num_limited:
        five_star = False
        if initial_pity_flag:
            count = initial_pity
            initial_pity_flag = False
        else:
            count = 0
        while not five_star:
            wish = rng.binomial(1, five_star_rate + prob_increase * max(count - soft_pity_thresh, 0))
            rarity = five_star_map[wish]
            if rarity == '5 Star':
                if last_five_star == 'Limited':
                    wish = rng.binomial(1, 0.5) if banner_type == 'Character' else rng.binomial(1, 0.75)
                else:
                    wish = 1
                rarity = fiftyfifty_map[wish]
                last_five_star = rarity
                limited += wish
                five_star = True
            count += 1
        wishes.append(count)
    # Adjusting for the initial pity
    wishes[0] = wishes[0] - initial_pity
    return sum(wishes)

@st.cache_data
def sim_two_limited(M, count, banner_type, last_five_star, num_limited):
    results = []
    if banner_type == 'Character' or 'Light Cone':
        for i in range(M):
            results.append(limited_wish(count, banner_type, last_five_star, num_limited))
    return results

@st.cache_data
def get_percentiles(results):
    probabilities = np.linspace(0.1, 0.9, 9)
    percentiles = np.quantile(results, probabilities)
    statistics = pd.DataFrame(data={'Warps': percentiles})
    statistics.index = list(map(lambda x: x+'%', (probabilities * 100).astype(int).astype(str)))
    statistics.index.name = 'Percentiles'
    return statistics

@st.cache_data
def get_descriptions(results):
    def percentile(n):
        def percentile_(x):
            return x.quantile(n)
        percentile_.__name__ = 'percentile_{:02.0f}'.format(n*100)
        return percentile_
    df = pd.DataFrame({
        'Warps': results,
    })
    agg = df.agg([np.mean, np.std, np.min, percentile(0.25), percentile(0.5), percentile(0.75), np.max])
    agg.index = ['Mean', 'Standard Devaiation', 'Min', 'First Quartile', 'Median', 'Third Quartile', 'Max']
    agg.index.name = 'Statistics'
    return agg.round(2)


st.title('Honkai Star Rail Warp Calculator')

with st.sidebar:
    about, methodology = st.tabs(['About', 'Methodology'])
    with about:
        st.markdown("""
                    Gives probability and statistics of getting limited 5 star characters, given 
                    initial pity state using 10000 simulations. 
        """)
    with methodology:
        st.write('Assumptions (Character/LC):')
        st.markdown("""
                    Five Star Rate: 0.6% / 0.8%  
                    Soft Pity Starts on: 74 / 63  
                    Soft pity increases linearly.
        """)
    st.header('Parameters')
    with st.form('Parameters'):
        num_limited = st.number_input('How many limited 5 stars do you want to pull for?', min_value=1, max_value=6)
        pity = st.number_input('Current Pity', min_value=0, max_value=89)
        banner_type = st.selectbox('Banner Type', ('Character', 'Light Cone'))
        state = st.toggle('Next 5 star a guaranteed limited?')
        last_five_star = 'Standard' if state else 'Limited'
        
        start_analysis = st.form_submit_button('Estimate')
    
    if start_analysis:
        st.cache_data.clear()

rng = np.random.default_rng()
results = np.array(sim_two_limited(10000, pity, banner_type, last_five_star, num_limited))

st.subheader('Warp Distribution')
hist = px.histogram(results, nbins=40)
hist.update_traces(hovertemplate ='<br>'.join([
    'Bin Range: %{x}',
    'Frequency: %{y}' 
]), selector=dict(type="histogram"))
hist.update_layout(showlegend=False, xaxis_title_text='Warps', yaxis_title_text='Frequency')
st.plotly_chart(hist)

st.subheader('Descriptive Statistics')
statistics = get_percentiles(results)
desc = get_descriptions(results)
col1, col2 = st.columns([0.33, 0.67])
with col1:
    st.dataframe(statistics, use_container_width=True)
with col2:
    st.dataframe(desc, use_container_width=True)

st.subheader('Probabilities')
ecdf= ECDF(results)
with st.form('Probailities'):
    col1, col2 = st.columns(2)
    with col1:
        low = st.number_input('Lower Bound for Warps', min_value=0, value=int(np.quantile(results, 0.25)))
    with col2:
        high = st.number_input('Upper Bound for Warps', min_value=0, value=int(np.quantile(results, 0.75)))
    if st.form_submit_button('Estimate'):
        p = ecdf(high) - ecdf(low)
        st.write(f'Probability: {p:.2%}')