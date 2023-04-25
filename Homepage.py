import streamlit as st

st.set_page_config(
    page_title="Hello Page",
    page_icon="📚",
    layout='wide'
)

st.title("📚 *GoodReads*: Book Analytics & Recommendation System")
st.markdown(
    """
    - @Group 18
    - @Member: Yanbo Wang, Yinuo Hu, Shichen Wu, Jiaming Xiong, Xinlin Huang, Wenxuan Yang
    - @GitHub: [GoodReads-Recommendation-System](https://github.com/YanboWang0204/GoodReads-Recommendation-System)
    ---
    
    This webpage app is built specifically for demonstrating our group project about
    building an intelligent book recommendation system based on data from *GoodReads.com*.
    
    **👈 You can switch different pages on left-hand side bar to play with different functions
    """
)

st.subheader('📊 Visualization & Exploratory Analysis')
st.markdown(
    """
    In this module, we conducted data visualization & exploratory data analysis.\
    
    Here we classify books by their genres and you can select genres you wanted in the multi-select boxes for 
    specific visualization.
    """
)

st.subheader('📜 Recommendation System')
st.markdown(
    """
    This is the main module for our recommendation system. You can try our system by following up 
    procedures described in pages.
    - Recommendation models
        - Popularity model (Baseline)
        - Content-based filtering
        - Collaborative filtering
        - Ensemble model
    
    We also explored some NLP models to provide some executive review of book content and reviews, 
    which can help users quickly grasp a screenshot of books recommended to them.
    """
)

st.subheader('📅 Future Plans')
st.markdown(
    """
    Based on our current work, we have deployed this app to help potential users to generate their unique
    to-read lists. As a next step, we will broader our database and pursue a higher recommendation accuracy to 
    provide better experiences for our users 
    """
)

st.subheader('🗒️ Documentation')
st.markdown(
    """
    Explanation of train of thoughts of project, including:
    - Data collection (web scraping paths)
    - Datasets uses
    - Recommendation models & NLP models
    """
)

st.subheader('📩 Contact')
st.markdown(
    """
    If you have any questions, please contact us by email: 📩 [yanbo.wang@berkeley.edu](yanbo.wang@berkeley.edu)
    """
)

st.markdown(
    """
    ---
    Version 1.0, copyright by Yanbo Wang (IEOR, UC Berkeley)
    """
)

# c1 = st.columns(1)
# c1.write()
# st.info('**GitHub: [@GoodReads-Recommendation-System](https://github.com/YanboWang0204/GoodReads-Recommendation-System)**', icon="💻")