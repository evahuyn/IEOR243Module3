import streamlit as st
import pandas as pd
import numpy as np
from ast import literal_eval
import time
import sys
sys.path.append("E:\\Files\\UCB\\Courses\\INDEND_243\\module3\\GoodReads_Interaction\\pages\\Models")

# import models classes
from Popularity import PopularityRecommender
from Content_based import ContentBasedRecommender, build_users_profiles
from Collaborative_Filtering import CFRecommender
from Hybrid import HybridRecommender

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Recommendation System",
    page_icon=":orange_book:",
    layout='wide'
)

st.title('📜 Recommendation System')
st.markdown(
    """
    ### Welcome to our recommendation engines!
    
    Here we will first request you to choose some book genres you are curious about and 
    provide a list of books from our database to collect your responses.
    
    Then our fine-tuned models will recommend books you might be interested in and provide you with 
    some snapshots of their detailed information. Even if you did not read any of these books, don't worry!
    We will recommend top famous books prevailing among our readers on *GoodReads.com*.
    
    ---
    """
)


@st.cache_resource
def load_data_google(sheets_url):
    csv_url = sheets_url.replace("/edit#gid=", "/export?format=csv&gid=")
    return pd.read_csv(csv_url)


@st.cache_data
def list_transform(df):
    df['Full_Genres'] = df['Full_Genres'].apply(lambda x: literal_eval(x) if "[" in x else x)
    df['Award'] = df['Award'].apply(lambda x: literal_eval(x) if "[" in x else x)
    return df


@st.cache_data
def list_transform_book(df):
    df['Genres'] = df['Genres'].apply(lambda x: literal_eval(x) if "[" in x else x)
    df['Award'] = df['Award'].apply(lambda x: literal_eval(x) if "[" in x else x)
    return df


@st.cache_data
def svd_pred(user_interactions_df):
    users_items_pivot_matrix_df = user_interactions_df.pivot(index='UserID',
                                                             columns='Uid',
                                                             values='Review_Rating').fillna(0)

    users_items_pivot_matrix = users_items_pivot_matrix_df.to_numpy()
    users_ids = list(users_items_pivot_matrix_df.index)

    users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)

    U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k=23)
    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / \
                                      (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())

    cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns=users_items_pivot_matrix_df.columns,
                               index=users_ids).transpose()

    return cf_preds_df


def progress_bar():
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(6):
        time.sleep(0.1)
        my_bar.progress(percent_complete * 20, text=progress_text)

    pass


# def button_callback():
#     if button_no_book not in st.session_state:
#         st.session_state.button_no_book = False
#         st.session_state.button_select = True


def display_bar():

    with st.container():
        book_interest = st.multiselect(
            "Do you have some books interested in? Choose them from the following box:",
            book_rec_df['Title']
        )

        for book in book_interest:
            with st.expander(book):
                book_interest_df = books_df[books_df['Title'] == book]
                uid = book_interest_df['Uid'].values[0]

                c_img, c_info = st.columns(2)
                with c_img:
                    pic_url = book_pic_url[book_pic_url['Uid'] == uid]['URL'].values[0]
                    st.image(pic_url, width=300)

                with c_info:
                    st.metric(label='Book name', value=book)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric(label='Author', value=book_interest_df['Author'].values[0])

                    c2, c3, c4 = st.columns(3)
                    with c2:
                        st.metric(label='Average Rating', value=book_interest_df['Rating'].values[0])
                    with c3:
                        st.metric(label='Rating Number', value=book_interest_df['Rating_Num'].values[0])
                    with c4:
                        st.metric(label='Review Number', value=book_interest_df['Review_Num'].values[0])

                    fig_RatingDist = px.bar(book_interest_df,
                                            x=["Five_star_percent", "Four_star_percent",
                                               "Three_star_percent", "Two_star_percent", "One_star_percent"],
                                            text_auto=True,
                                            width=550, height=200)

                    fig_RatingDist.update_layout(yaxis={'visible': False, 'showticklabels': False},
                                                 title_text="Rating Distribution",
                                                 xaxis_title='Percentage',
                                                 title_x=0,
                                                 uniformtext_minsize=8, uniformtext_mode='hide')

                    st.plotly_chart(fig_RatingDist)

                # plot book stats
                stats_plot(book_interest_df)

                tag_list = book_tags[book_tags['Uid'] == uid]['Tags']
                book_sum = book_summary[book_summary['Uid'] == uid]

                st.markdown(
                    """
                    #### Book tags
                    """
                )
                # tag_str = ' '.join(tag_list)
                st.dataframe(tag_list)
                # st.write(tag_str)

                st.markdown(
                    """
                    #### Book summary
                    """
                )
                book_sum_df = book_summary_process(book_sum)
                st.dataframe(book_sum_df, width=1200)

                url = 'https://www.goodreads.com/book/show/' + str(book_interest_df['Uid'].values[0])
                st.write("Book URL: " + url)

    pass


@st.cache_data
def stats_plot(book_df):
    uid = book_df['Uid'].values[0]
    df_stats = book_stats[book_stats['Uid'] == uid]

    st.markdown(
        """
        #### Book statistics
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(label="Added", value=df_stats.iloc[0, 1])

    with c2:
        st.metric(label="Ratings", value=df_stats.iloc[0, 2])

    with c3:
        st.metric(label="Reviews", value=df_stats.iloc[0, 3])

    with c4:
        st.metric(label="To-read", value=df_stats.iloc[0, 4])

    df_stats_day = df_stats.iloc[1:, :]

    c1, c2 = st.columns(2)
    with c1:
        p1 = go.Figure()
        p1.add_trace(go.Scatter(x=df_stats_day['date'].values,
                                y=df_stats_day['added'].values,
                                mode='lines+markers',
                                name='Added',
                                line=dict(color='firebrick', width=2)))
        p1.update_layout(width=400, height=300,
                         title_text='Added', title_x=0.4)
        st.plotly_chart(p1)

    with c2:
        p2 = go.Figure()
        p2.add_trace(go.Scatter(x=df_stats_day['date'].values,
                                y=df_stats_day['ratings'].values,
                                mode='lines+markers',
                                name='Ratings',
                                line=dict(color='orange', width=2)))
        p2.update_layout(width=400, height=300,
                         title_text='Ratings', title_x=0.4)
        st.plotly_chart(p2)

    c3, c4 = st.columns(2)
    with c3:
        p3 = go.Figure()
        p3.add_trace(go.Scatter(x=df_stats_day['date'].values,
                                y=df_stats_day['reviews'].values,
                                mode='markers',
                                name='Reviews',
                                line=dict(color='royalblue', width=2, dash='dot')))
        p3.update_layout(width=400, height=300,
                         title_text='Reviews', title_x=0.4)
        st.plotly_chart(p3)

    with c4:
        p4 = go.Figure()
        p4.add_trace(go.Scatter(x=df_stats_day['date'].values,
                                y=df_stats_day['to-read'].values,
                                mode='lines+markers',
                                name='to-read',
                                line=dict(color='green', width=2)))
        p4.update_layout(width=400, height=300,
                         title_text='To-read', title_x=0.4)
        st.plotly_chart(p4)

    pass


def book_summary_process(book_sum):
    summary = book_sum['Summary'].values[0]
    sum_split = summary.split('[SEP]')[:-1]
    return pd.DataFrame(sum_split, columns=['Summary'])


# load data

df_info = load_data_google('https://docs.google.com/spreadsheets/d/1pKiV2vD3ZTvjqg0EYyYa9-3whnVaFhGQoYQCHlk7GRE/edit#gid=1668223231')
df_info = list_transform(df_info)

books_df = load_data_google('https://docs.google.com/spreadsheets/d/1lt1VLkm728Lkx9zJV7XLmui6_qSirNOHWnHVIqALL4U/edit#gid=1540274189')
books_df = list_transform_book(books_df)

interactions_df = load_data_google('https://docs.google.com/spreadsheets/d/188-QkQltO5e-dsQHFozPS7PeuKjInO1q-g3aJ-1fmy0/edit#gid=1215915440')

# book tags & review summary
book_tags = load_data_google('https://docs.google.com/spreadsheets/d/1441yNTGqVkC_MoFlVK2d_XNJP97nxjSDo8nBUvC0NA8/edit#gid=1641136196')
book_summary = load_data_google('https://docs.google.com/spreadsheets/d/1NXfBMQ1w9qrknZuKxGQHqPVHcmV1-8LfpDc930iI-Kk/edit#gid=193127644')[['Uid', 'Summary']]

# book picture URLs
book_pic_url = load_data_google('https://docs.google.com/spreadsheets/d/10f6yOXebLT4tWrevev-UH4TgHfxx9gL9cwmOn3iaJUg/edit#gid=65438505')

# book stats
book_stats = load_data_google('https://docs.google.com/spreadsheets/d/1Oxbmc_OP3GTwPiXhYyoXUVK-pEgancDejCpBFHlF6bc/edit#gid=726264491')


genre_selected = st.multiselect(
                        "Hi, please choose the genre you are curious about: ",
                        set(df_info['Genre']),
                        ['fiction', 'art']
)

df_select = df_info[df_info['Genre'].isin(genre_selected)]

book_selected = st.multiselect(
        "Choose books you have read and rate them from 1 (bad) to 5 (best)",
        df_select['Title'].unique()
)

if not book_selected:

    button_no_book = st.button("No book I have read before")
    if 'button_no_book' not in st.session_state:
        st.session_state.button_no_book = False

    # if button_no_book:
    #     st.session_state.button_no_book = True
    #
    # if st.session_state.button_no_book:
    #     st.write("If no books you have read before, please take a look at the top-rating books in your curious genres:")
    #
    #     item_popularity_df = interactions_df.groupby('Uid')['Review_Rating'].sum().sort_values(
    #         ascending=False).reset_index()
    #
    #     popularity_model = PopularityRecommender(item_popularity_df, df_select)
    #     recommend_df = popularity_model.recommend_items()
    #
    #     book_rec_uid = recommend_df['Uid'].values
    #     book_rec_df = books_df[books_df['Uid'].isin(book_rec_uid)]
    #     st.dataframe(book_rec_df[['Title', 'Rating']])
    #
    #     with st.form("No book"):
    #         display_bar()
    #         st.form_submit_button("Submit Form")
    # if button_no_book:
    #     try:
    #         st.session_state.button_no_book = False
    #     except AttributeError:
    #         if 'button_no_book' not in st.session_state:
    #             st.session_state.button_no_book = True
    #     else:
    #         st.session_state.button_no_book = True

    if button_no_book:
        st.session_state.button_no_book = True

    if st.session_state.button_no_book:
        st.write("If no books you have read before, please take a look at the top-rating books in your curious genres:")

        item_popularity_df = interactions_df.groupby('Uid')['Review_Rating'].sum().sort_values(
            ascending=False).reset_index()

        popularity_model = PopularityRecommender(item_popularity_df, df_select)
        recommend_df = popularity_model.recommend_items()

        book_rec_uid = recommend_df['Uid'].values
        book_rec_df = books_df[books_df['Uid'].isin(book_rec_uid)]
        st.dataframe(book_rec_df[['Title', 'Rating']])

        # st.session_state.button_no_book = True
        display_bar()


else:
    # add new user info
    uid = df_select[df_select['Title'].isin(book_selected)]['Uid'].values
    n = len(book_selected)
    new_user_rating = []
    user_ID = [uid for uid in [13223456] for i in range(n)]

    for name in book_selected:
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label="Book name", value=name)

        with c2:
            rating = st.select_slider(
                "What's your rating for this book?",
                options=[1, 2, 3, 4, 5],
                key=name
            )
            new_user_rating.append(rating)

    # Content-based
    new_user_df = pd.DataFrame({"Uid": uid,
                                "UserID": user_ID,
                                "Review_Rating": new_user_rating})

    model = st.radio(
        "Choose one recommendation engine for you:",
        ('Content-based', 'Collaborative-Filtering', 'Hybrid')
    )

    button = st.button("Run")
    if 'button' not in st.session_state:
        st.session_state.button = False

    if model == 'Content-based':
        new_user_profile = build_users_profiles(new_user_df)

        content_based_recommender_model = ContentBasedRecommender(books_df)
        recommend_df = content_based_recommender_model.recommend_items(user_id=13223456, user_profile=new_user_profile,
                                                                       items_to_ignore=uid)
        book_rec_uid = recommend_df['Uid'].values
        book_rec_df = books_df[books_df['Uid'].isin(book_rec_uid)]

        if button:
            st.session_state.button = True

        if st.session_state.button:
            progress_bar()
            st.write("Based on your ratings, we guess the following books might be your favourites!")
            st.dataframe(book_rec_df[['Title', 'Rating']])
            st.balloons()

            # st.session_state.button = True
            display_bar()

    if model == 'Collaborative-Filtering':
        # Collaborative filtering

        total_interactions_df = pd.concat([interactions_df[['Uid', 'UserID', 'Review_Rating']], new_user_df], axis=0)

        cf_preds_df = svd_pred(total_interactions_df)

        cf_recommender_model = CFRecommender(cf_preds_df, books_df)
        book_rec_cf = cf_recommender_model.recommend_items(user_id=13223456, items_to_ignore=uid)

        book_cf_uid = book_rec_cf['Uid'].values
        book_rec_df = books_df[books_df['Uid'].isin(book_cf_uid)]

        if button:
            st.session_state.button = True

        if st.session_state.button:
            progress_bar()
            st.write("Based on users who have similar tastes as you,  we guess the following books might be your favourites!")
            st.dataframe(book_rec_df[['Title', 'Rating']])
            st.balloons()

            # st.session_state.button = True
            display_bar()

    if model == 'Hybrid':
        # Hybrid models
        new_user_profile = build_users_profiles(new_user_df)
        content_based_recommender_model = ContentBasedRecommender(books_df)

        total_interactions_df = pd.concat([interactions_df[['Uid', 'UserID', 'Review_Rating']], new_user_df], axis=0)
        cf_preds_df = svd_pred(total_interactions_df)
        cf_recommender_model = CFRecommender(cf_preds_df, books_df)

        hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, books_df,
                                                     cb_ensemble_weight=1.0, cf_ensemble_weight=1.0)

        book_rec_hybrid = hybrid_recommender_model.recommend_items(user_id=13223456, user_profile=new_user_profile,
                                                                   items_to_ignore=uid)

        book_hybrid_uid = book_rec_hybrid['Uid'].values
        book_rec_df = books_df[books_df['Uid'].isin(book_hybrid_uid)]

        if button:
            st.session_state.button = True

        if st.session_state.button:
            progress_bar()
            st.write("Based on your ratings & other similar users' tastes, we guess the following books might be your favourites!")
            st.dataframe(book_rec_df[['Title', 'Rating']])
            st.balloons()

            # st.session_state.button = True
            display_bar()

        # if button or st.session_state.button:
        #     progress_bar()
        #     st.write("Based on your ratings & other similar users' tastes, we guess the following books might be your favourites!")
        #     st.dataframe(book_rec_df[['Title', 'Rating']])
        #     st.balloons()
        #
        #     st.session_state.button = True
        #     display_bar()


