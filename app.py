import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from PIL import Image
#from transformers import GPT2LMHeadModel, GPT2Tokenizer
import requests
import json

# Load CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Display images
col1, col2, col3 = st.columns(3)

with col1:
    #st.header("By")
    img1 = Image.open('Logo.png')
    st.image(img1, width=150)

st.markdown("""
    <style>
        .big-font {
            font-size:25px !important;
            color: white;
            font-family: 'Trebuchet MS';
            text-transform: uppercase;  # Transform text to uppercase
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);  # Add text shadow
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">AI enabled investor decision system for restaurants</p>', unsafe_allow_html=True)

# with col2:
#     st.header("For")
#     img2 = Image.open('event.png')
#     st.image(img2, width=150)

with col3:
    # st.header("Powered by")
    img3 = Image.open('canvas.png')
    st.image(img3, width=150)


    # st.header("Tech used")
    # img4 = Image.open('streamlit.png')
    # st.image(img4, width=150)

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Submit button
if st.button("Submit"):
    if uploaded_file is not None:
        # Load the dataset
        data = pd.read_csv(uploaded_file)

        # Data Preprocessing
        categorical_features = ['Location', 'Cuisine', 'Parking Availability']
        numerical_features = data.columns.difference(categorical_features + ['Name', 'Revenue'])

        # Preprocessing pipeline for numerical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Preprocessing pipeline for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Apply preprocessing
        data_processed = preprocessor.fit_transform(data)

        # Convert the processed data back to a DataFrame
        data_processed_df = pd.DataFrame(data_processed, columns=numerical_features.tolist() + preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features).tolist())

        # Add the 'Name' and 'Revenue' columns back
        data_processed_df['Name'] = data['Name']
        data_processed_df['Revenue'] = data['Revenue']

        # Calculate Investment Score
        # Normalize relevant columns
        columns_to_normalize = ['Revenue', 'Rating', 'Seating Capacity', 'Average Meal Price', 'Marketing Budget', 'Social Media Followers', 'Chef Experience Years', 'Number of Reviews', 'Avg Review Length', 'Ambience Score', 'Service Quality Score', 'Weekend Reservations', 'Weekday Reservations']
        scaler = StandardScaler()
        data_processed_df[columns_to_normalize] = scaler.fit_transform(data_processed_df[columns_to_normalize])

        # Define weights for each factor
        weights = {
            'Revenue': 0.2,
            'Rating': 0.15,
            'Seating Capacity': 0.1,
            'Average Meal Price': 0.05,
            'Marketing Budget': 0.1,
            'Social Media Followers': 0.05,
            'Chef Experience Years': 0.1,
            'Number of Reviews': 0.05,
            'Avg Review Length': 0.05,
            'Ambience Score': 0.05,
            'Service Quality Score': 0.05,
            'Weekend Reservations': 0.025,
            'Weekday Reservations': 0.025
        }

        # Calculate the investment score
        data_processed_df['Investment Score'] = sum(data_processed_df[col] * weight for col, weight in weights.items())

        # Rank Restaurants
        data_processed_df = data_processed_df.sort_values(by='Investment Score', ascending=False)

        # Identify Top 10 and Bottom 10 Restaurants
        top_10_invest = data_processed_df.head(10)
        bottom_10_invest = data_processed_df.tail(10)

        # Visualization
        st.subheader("Top 10 Restaurants for Investment Based on Initial Score")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='Investment Score', y='Name', data=top_10_invest, palette='viridis', ax=ax)
        ax.set_title('Top 10 Restaurants for Investment Based on Initial Score')
        ax.set_xlabel('Investment Score')
        ax.set_ylabel('Restaurant')
        st.pyplot(fig)

        st.subheader("Bottom 10 Restaurants for Investment Based on Initial Score")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='Investment Score', y='Name', data=bottom_10_invest, palette='magma', ax=ax)
        ax.set_title('Bottom 10 Restaurants for Investment Based on Initial Score')
        ax.set_xlabel('Investment Score')
        ax.set_ylabel('Restaurant')
        st.pyplot(fig)

        # Define the model
        model = GradientBoostingRegressor(n_estimators=100, random_state=0)

        # Create and evaluate the pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)
                                  ])

        # Split the data into training and testing sets
        X = data.drop('Revenue', axis=1)
        y = data['Revenue']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        pipeline.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = pipeline.predict(X_test)
        st.write('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
        st.write('Mean Squared Error:', mean_squared_error(y_test, y_pred))
        st.write('R-squared:', r2_score(y_test, y_pred))

        # Output the top 10 predictions
        predictions_df = pd.DataFrame({'Actual Revenue': y_test, 'Predicted Revenue': y_pred})
        top_10_predictions = predictions_df.nlargest(10, 'Predicted Revenue')
        st.write("Top 10 Predicted Revenues:")
        st.write(top_10_predictions)

        # Visualization of Actual vs Predicted Revenue for Top 10 Predicted Revenues
        st.subheader("Actual vs Predicted Revenue for Top 10 Predicted Revenues")
        fig, ax = plt.subplots(figsize=(12, 6))
        top_10_predictions.plot(kind='bar', ax=ax)
        ax.set_title('Actual vs Predicted Revenue for Top 10 Predicted Revenues')
        ax.set_xlabel('Restaurant Index')
        ax.set_ylabel('Revenue')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)

        # To visualize the residuals (differences between actual and predicted values) and check for patterns that might indicate model issues.
        # Residual Plot
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=residuals, ax=ax)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        st.pyplot(fig)

        # Actual vs Predicted Values
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs Predicted Values')
        st.pyplot(fig)

        # To visualize the distribution of investment scores.
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data_processed_df['Investment Score'], kde=True)
        ax.set_title('Distribution of Investment Scores')
        ax.set_xlabel('Investment Score')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        # # To show the importance of each feature in the model.
        # importances = model.feature_importances_
        # feature_names = X.columns
        # feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        # feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # fig, ax = plt.subplots(figsize=(10, 6))
        # sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        # ax.set_title('Feature Importance')
        # ax.set_xlabel('Feature')
        # ax.set_ylabel('Importance')
        # st.pyplot(fig)

        # Get feature importances from the model
        # importances = model.feature_importances_

        # # Get feature names from the preprocessor
        # numerical_features = X.columns.difference(categorical_features + ['Name', 'Revenue'])
        # categorical_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
        # feature_names = numerical_features.tolist() + categorical_feature_names.tolist()

        # # Check if the lengths of 'importances' and 'feature_names' are the same
        # if len(importances) == len(feature_names):
        #     # Create a DataFrame to display feature importances
        #     feature_importance_df = pd.DataFrame({
        #         'Feature': feature_names,
        #         'Importance': importances
        #     })
        #     feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        #     # Plot the feature importances
        #     fig, ax = plt.subplots(figsize=(10, 6))
        #     sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        #     ax.set_title('Feature Importance')
        #     ax.set_xlabel('Feature')
        #     ax.set_ylabel('Importance')
        #     st.pyplot(fig)
        # else:
        #     print("Error: The length of 'importances' and 'feature_names' does not match.")

        st.subheader("Top 10 Predicted Revenues")
        for index, row in top_10_predictions.iterrows():
            restaurant_info = X_test.loc[index].to_dict()
            input_text = (
                f"The restaurant with actual revenue of {row['Actual Revenue']} and predicted revenue of {row['Predicted Revenue']} "
                f"is located in {restaurant_info['Location']} and serves {restaurant_info['Cuisine']} cuisine. It has a rating of {restaurant_info['Rating']} "
                f"and a seating capacity of {restaurant_info['Seating Capacity']}. The average meal price is {restaurant_info['Average Meal Price']} and "
                f"it has a marketing budget of {restaurant_info['Marketing Budget']}. With {restaurant_info['Social Media Followers']} social media followers, "
                f"the head chef has {restaurant_info['Chef Experience Years']} years of experience. The restaurant has received {restaurant_info['Number of Reviews']} reviews "
                f"with an average length of {restaurant_info['Avg Review Length']}. The ambience score is {restaurant_info['Ambience Score']} and the service quality score is "
                f"{restaurant_info['Service Quality Score']}. Parking availability: {restaurant_info['Parking Availability']}. The restaurant makes {restaurant_info['Weekend Reservations']} weekend reservations and "
                f"{restaurant_info['Weekday Reservations']} weekday reservations. Given these characteristics, the restaurant should consider "
                f"focusing on improving {('service quality' if restaurant_info['Service Quality Score'] < 3 else 'ambience' if restaurant_info['Ambience Score'] < 3 else 'social media engagement')} to boost revenue."
            )
            st.write(f"Insight for Restaurant {index}:\n{input_text}\n")

        def generate_insight(query):
            try:
                # Define the URL
                url = "https://canvasai.ltimindtree.com/chatservice/chat"
                # Define the headers
                headers = {
                    'accept': 'application/json',
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJmdVdXcndULWRia28xZ2JwbmFFWkU2V2d1RjJ4RERGSHAyMlYzdUV6UDA4In0.eyJleHAiOjE3MzY3NTgzNzMsImlhdCI6MTczNjc1Njg3MywiYXV0aF90aW1lIjoxNzM2NzU2ODcxLCJqdGkiOiIwYzcyZDM2NC03YWUzLTQ0ZmMtYWNmOS1mZDBiNTZhOGI1OTciLCJpc3MiOiJodHRwczovL2NhbnZhc2FpLmx0aW1pbmR0cmVlLmNvbS9rZXljbG9hay9yZWFsbXMvY2FudmFzYWkiLCJhdWQiOlsiYnJva2VyIiwiYWNjb3VudCJdLCJzdWIiOiIzZWZmODllMS1mY2I1LTQ3MmEtYjdiNC1mMTNjMWQwNzYwM2YiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJjYW52YXNhaS1jb250cm9scGxhbmUtdWkiLCJub25jZSI6ImIzODI3NDYyLWNkNzAtNGZmNy04OTA3LTJhN2MwNDc2ZmI0YSIsInNlc3Npb25fc3RhdGUiOiI5ZGVhOWQ1OC01ZTQzLTRmOWQtYjVjYS0zNjliZTFmM2VkYjIiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbImh0dHBzOi8vY2FudmFzYWkubHRpbWluZHRyZWUuY29tL3N0dWRpbyIsIioiXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbImRlZmF1bHQtcm9sZXMtY2FudmFzYWkiLCJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYnJva2VyIjp7InJvbGVzIjpbInJlYWQtdG9rZW4iXX0sImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoib3BlbmlkIGVtYWlsIEdyb3VwU2NvcGUgcHJvZmlsZSIsInNpZCI6IjlkZWE5ZDU4LTVlNDMtNGY5ZC1iNWNhLTM2OWJlMWYzZWRiMiIsInVwbiI6ImQwODRmYmE2LTUwNTktNGRkMy04NzkxLTJmZGI2NjAwYzJjOCIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJuYW1lIjoiUmFodWwgQmhhdmUiLCJncm91cHMiOlsiZGVmYXVsdC1yb2xlcy1jYW52YXNhaSIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXSwicHJlZmVycmVkX3VzZXJuYW1lIjoiZDA4NGZiYTYtNTA1OS00ZGQzLTg3OTEtMmZkYjY2MDBjMmM4IiwiZ2l2ZW5fbmFtZSI6IlJhaHVsIiwiZmFtaWx5X25hbWUiOiJCaGF2ZSIsImVtYWlsIjoicmFodWwuYmhhdmVAbHRpbWluZHRyZWUuY29tIn0.V-C5nXcoox63c1Tg0_urmWQQmMm0jTveDT7vLTfxyfepT_El28sOdinm-XwdqF9ZgFntASdEK1N1ulZ7G4CPwG2eLz3oFQpn6otzKXEbHeAkeGxdJKYHLgL20R_9QgdUwpyM5yEkQnf3nQ8wmBQ0rkL_jkl7AwjGAxguBcLsmqkMRiz74Tjyof1tm0ZwddPClmQdguL9Qxzo_PFLb2eOF4lorVybe-yg9UkqSRg-RCigR9B2LVPYzyJul7tE7OIxS2WN9W-R1ulD-lQ5eY7lTwoVEG8bg4haoh_k-hxRYR16rgRQT7sB4o2XRRSXfezG5CjX-Pex2t7sA9U7p5rD0g'
                }
                # Define the data
                data = {
                    "hint": "You are an expert restaurant inestement intelligent assistant.",
                    "query": query,
                    "space_name": "snowspark",
                    "userId": "d084fba6-5059-4dd3-8791-2fdb6600c2c8"
                }
                # Make the POST request
                response = requests.post(url, headers=headers, data=json.dumps(data))
                # Return the response
                parsed_response = response.json()
                return parsed_response['response']
            except Exception:
                return f"Either your tokens got expired or Canvas API is not available, please try after sometime"

        # issue with Top 10 predicted Revenues
        # st.subheader("Insights for Top 10 Predicted Revenues")
        # for index, row in top_10_predictions.iterrows():
        #     restaurant_info = X_test.loc[index].to_dict()
        #     if restaurant_info['Service Quality Score'] > 9 and restaurant_info['Rating'] > 4:
        #         insight = "Given these characteristics, since the restaurant has service quality greater than 9 and rating greater than 4 we should consider investing in this restaurant."
        #     else:
        #         input_text = (
        #             f"As an intelligent investment assistant, please study the following insights and share your insights:\n"
        #             f"Case 1: The restaurant with actual revenue of 1482906.08 and predicted revenue of 1467845.1875506877 is located in Downtown and serves Japanese cuisine. It has a rating of 4.2 and a seating capacity of 90. The average meal price is 73.64 and it has a marketing budget of 5305. With 55378 social media followers, the head chef has 4 years of experience. The restaurant has received 237 reviews with an average length of 236.68326845972933. The ambience score is 8.4 and the service quality score is 9.1. Parking availability: No. The restaurant makes 31 weekend reservations and 71 weekday reservations.Insight:Given these characteristics, since the restaurant has service quality greater than 9 and rating greater than 4 we should consider investing in restaurant.\n"
        #             f"Case 2: The restaurant with actual revenue of 1466609.38 and predicted revenue of 1465533.2835080354 is located in Downtown and serves Japanese cuisine. It has a rating of 3.1 and a seating capacity of 87. The average meal price is 75.41 and it has a marketing budget of 5262. With 58295 social media followers, the head chef has 9 years of experience. The restaurant has received 589 reviews with an average length of 100.43742175956096. The ambience score is 1.3 and the service quality score is 2.8. Parking availability: No. The restaurant makes 81 weekend reservations and 9 weekday reservations.Insight:since service quality is less than 9 and rating is also less than 4, we should not invest in this restaurant, to increase investibility this restaurant should improve aspects.\n"
        #             f"After carefully analysing above 2 cases, would you be able to generate insight for case 4, please share insights for below cases:\n"
        #             f"Case {index}: The restaurant with actual revenue of {row['Actual Revenue']}  and predicted revenue of {row['Predicted Revenue']} is located in {restaurant_info['Location']} and serves {restaurant_info['Cuisine']} cuisine. It has a rating of {restaurant_info['Rating']} and a seating capacity of {restaurant_info['Seating Capacity']}. The average meal price is {restaurant_info['Average Meal Price']} and it has a marketing budget of {restaurant_info['Marketing Budget']}. With {restaurant_info['Social Media Followers']} social media followers, the head chef has {restaurant_info['Chef Experience Years']} years of experience. The restaurant has received {restaurant_info['Number of Reviews']} reviews with an average length of {restaurant_info['Avg Review Length']}. The ambience score is {restaurant_info['Ambience Score']} and the service quality score is {restaurant_info['Service Quality Score']}. Parking availability: {restaurant_info['Parking Availability']}. The restaurant makes {restaurant_info['Weekend Reservations']} weekend reservations and {restaurant_info['Weekday Reservations']} weekday reservations. Should we invest in this restaurant?"
        #         )
        #         insight = generate_insight(input_text)
        #     st.write(f"Insight for Restaurant {index}:\n{insight}\n")
        # else:
        #     st.error("Please upload a CSV file.")

        # st.subheader("Insights for Restaurants using Canvas")
        # for index, row in top_10_predictions.iterrows():
        #     restaurant_info = X_test.loc[index].to_dict()
        #     input_text = (
        #     f"As an expert Restaurant investment analyst, would you invest in the restaurant with a rating of {restaurant_info['Rating']} and a service quality score of {restaurant_info['Service Quality Score']} if they are greater than 4 and 9 respectively?\n")
        #     if restaurant_info['Service Quality Score'] > 9 and restaurant_info['Rating'] > 4:
        #         insight = "Given these characteristics, since the restaurant has service quality greater than 9 and rating greater than 4 we should consider investing in this restaurant."
        #     else:
        #         input_text = (
        #             f"Case 1: The restaurant with actual revenue of 1482906.08 and predicted revenue of 1467845.1875506877 is located in Downtown and serves Japanese cuisine. It has a rating of 4.2 and a seating capacity of 90. The average meal price is 73.64 and it has a marketing budget of 5305. With 55378 social media followers, the head chef has 4 years of experience. The restaurant has received 237 reviews with an average length of 236.68326845972933. The ambience score is 8.4 and the service quality score is 9.1. Parking availability: No. The restaurant makes 31 weekend reservations and 71 weekday reservations.Insight:Given these characteristics, since the restaurant has service quality greater than 9 and rating greater than 4 we should consider investing in restaurant.\n"
        #             f"Case 2: The restaurant with actual revenue of 1466609.38 and predicted revenue of 1465533.2835080354 is located in Downtown and serves Japanese cuisine. It has a rating of 3.1 and a seating capacity of 87. The average meal price is 75.41 and it has a marketing budget of 5262. With 58295 social media followers, the head chef has 9 years of experience. The restaurant has received 589 reviews with an average length of 100.43742175956096. The ambience score is 1.3 and the service quality score is 2.8. Parking availability: No. The restaurant makes 81 weekend reservations and 9 weekday reservations.Insight:since service quality is less than 9 and rating is also less than 4, we should not invest in this restaurant, to increase investibility this restaurant should improve aspects.\n"
        #             f"After carefully analysing above cases, would you be able to generate insight for case 4, please share insights for below cases:\n"
        #             f"case 4:The restaurant with actual revenue of 1458689.89 and predicted revenue of 1456727.3363985096 is located in Downtown and serves Japanese cuisine. It has a rating of 3.9 and a seating capacity of 87. The average meal price is 74.58 and it has a marketing budget of 4725. With 54753 social media followers, the head chef has 11 years of experience. The restaurant has received 992 reviews with an average length of 231.71139361853173. The ambience score is 8.1 and the service quality score is 6.8. Parking availability: Yes. The restaurant makes 15 weekend reservations and 85 weekday reservations. should we invest in this restaurant?"
        #         )
        #         # input_text = (
        #         #     f"As an expert Restaurant investment analyst, would you invest in the restaurant with Google rating 3.9 and service quality 8.1?\n"
        #         # )
        # insight_from_canvas = generate_insight(input_text)
        # print(insight_from_canvas)
        # st.write(f"Insight for Restaurant :\n{insight_from_canvas}\n")

        st.subheader("Selected Restaurants using LLM's")

        for index, row in top_10_predictions.iterrows():
            restaurant_info = X_test.loc[index].to_dict()
            print(restaurant_info['Rating'])
            input_text = (
                f"As an expert Restaurant investment analyst, would you invest in the restaurant with a rating of {restaurant_info['Rating']} and a service quality score of {restaurant_info['Service Quality Score']} if they are greater than 4 and 9 respectively?\n"
            )
            if restaurant_info['Service Quality Score'] > 9 and restaurant_info['Rating'] > 4:
                insight = "Given these characteristics, since the restaurant has service quality greater than 9 and rating greater than 4 we should consider investing in this restaurant."
            else:
                input_text = (
                    f"The restaurant {restaurant_info['Name']} is located in {restaurant_info['Location']} and serves {restaurant_info['Cuisine']} cuisine. It has a rating of {restaurant_info['Rating']} and a seating capacity of {restaurant_info['Seating Capacity']}. The average meal price is {restaurant_info['Average Meal Price']} and it has a marketing budget of {restaurant_info['Marketing Budget']}. With {restaurant_info['Social Media Followers']} social media followers, the head chef has {restaurant_info['Chef Experience Years']} years of experience. The restaurant has received {restaurant_info['Number of Reviews']} reviews with an average length of {restaurant_info['Avg Review Length']}. The ambience score is {restaurant_info['Ambience Score']} and the service quality score is {restaurant_info['Service Quality Score']}. Parking availability: {restaurant_info['Parking Availability']}. The restaurant makes {restaurant_info['Weekend Reservations']} weekend reservations and {restaurant_info['Weekday Reservations']} weekday reservations. Insight: Since service quality is less than 9 and rating is also less than 4, we should not invest in this restaurant. To increase investibility, this restaurant should improve aspects.\n"
                )
        insight_from_canvas = generate_insight(input_text)
        print(insight_from_canvas)
        st.write(f"Selected Restaurants using LLM's:\n{insight_from_canvas}\n")


