import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from statsmodels.distributions.empirical_distribution import ECDF


gem_cost = 160

def wish(initial_pity, last_five_star, num_limited):
    wishes = []
    limited = 0
    soft_pity_thresh = 73
    guarantee = 90
    five_star_map = {0: 'Not 5 Star', 1: '5 Star'}
    five_star_rate = 0.006
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
                    wish = rng.binomial(1, 0.5)
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
def sim_two_limited(M, count, last_five_star, num_limited):
    results = []
    for i in range(M):
        results.append(wish(count, last_five_star, num_limited))
    return results

@st.cache_data
def get_percentiles(results):
    probabilities = np.linspace(0.1, 0.9, 9)
    percentiles = np.quantile(results, probabilities)
    gem_percentiles = gem_cost * percentiles
    statistics = pd.DataFrame(data={'Pull Percentile': percentiles, 'Gem Percentiles': gem_percentiles})
    statistics.index = probabilities
    statistics.index.name = 'Probabilities'
    return statistics

@st.cache_data
def get_descriptions(results):
    def percentile(n):
        def percentile_(x):
            return x.quantile(n)
        percentile_.__name__ = 'percentile_{:02.0f}'.format(n*100)
        return percentile_
    df = pd.DataFrame({
        'Pulls': results,
        'Gems': results * gem_cost
    })
    agg = df.agg([np.mean, np.std, np.min, percentile(0.25), percentile(0.5), percentile(0.75), np.max])
    agg.index = ['Mean', 'Standard Devaiation', 'Min', 'First Quartile', 'Median', 'Third Quartile', 'Max']
    agg.index.name = 'Statistics'
    return agg.round(2)


st.title('Honkai Star Rail Warp Statistics')

with st.sidebar:
    st.header('About')
    st.markdown("""
                Gives probability and statistics of getting limited 5 star characters, given 
                initial pity state using 10000 simulations. Although I don't know about the 
                true probabilities of soft pity, I make an assumption that it increases linearly, 
                which holds up with the info we have, which is that expected pulls for a limited
                5 star character being 62.5.     
    """)
    st.write('Future Plans: Probability estimation for a range of pulls and different banner types.')
    st.header('Parameters')
    with st.form('Parameters'):
        num_limited = st.number_input('How many limited characters do you want to pull for?', min_value=1, max_value=6)
        pity = st.number_input('Current Pity', min_value=0, max_value=89)
        state = st.toggle('Next 5 star a guaranteed limited?')
        last_five_star = 'Standard' if state else 'Limited'
        
        start_analysis = st.form_submit_button('Estimate')
    
    if start_analysis:
        st.cache_data.clear()


rng = np.random.default_rng()
results = np.array(sim_two_limited(10000, pity, last_five_star, num_limited))

st.header('Statistics')

st.subheader('Pull Distribution')
hist = px.histogram(results, nbins=30)
hist.update_traces(hovertemplate ='<br>'.join([
    'Bin Range: %{x}',
    'Frequency: %{y}' 
]), selector=dict(type="histogram"))
hist.update_layout(showlegend=False, xaxis_title_text='Pulls', yaxis_title_text='Frequency')
st.plotly_chart(hist)

st.subheader('Descriptive Statistics')
statistics = get_percentiles(results)
desc = get_descriptions(results)
st.dataframe(statistics, use_container_width=True)
st.dataframe(desc, use_container_width=True)

st.subheader('Probabilities')
ecdf= ECDF(results)
with st.form('Probailities'):
    col1, col2 = st.columns(2)
    with col1:
        low = st.number_input('Lower Bound for Pulls', min_value=0, value=int(np.quantile(results, 0.25)))
    with col2:
        high = st.number_input('Upper Bound for Pulls', min_value=0, value=int(np.quantile(results, 0.75)))
    if st.form_submit_button('Estimate'):
        st.write(f'Probability: {round(ecdf(high) - ecdf(low), 4)}')