# Urban-Metro-Station-Site-Prediction
To analyze and predict metro stop locations across Indian cities.
**Project Overview:**
Urban Navigator is an ongoing initiative aimed at revolutionizing the planning and optimization of metro infrastructure in Indian cities. As urbanization accelerates, efficient public transportation becomes increasingly critical. This project leverages advanced data science and machine learning techniques to provide actionable insights into where new metro stops should be placed to best serve the population.

The core of Urban Navigator is a robust data-driven approach:

*Data Acquisition:* We gathered and compiled diverse datasets, including population density, proximity to popular destinations, hospitals, and traffic patterns. This data was meticulously curated through a combination of manual compilation and sophisticated web scraping techniques, ensuring that all relevant factors are considered in the analysis.

*City-Specific Datasets:* These datasets are then used to create city-specific profiles that include target columns indicating the presence or absence of metro stops at precise geographic coordinates.

*Machine Learning Models:* Using these datasets, we developed predictive models employing logistic regression, Random Forest, and XGBoost. These models analyze the compiled data to forecast the most effective locations for new metro stops.

**Technologies Used:**<br />
Python<br />
Tkinter (for any UI components)<br />
SQLite3 (for database management)<br />
Pandas, NumPy (for data manipulation)<br />
Scikit-learn (for machine learning models)<br />
XGBoost<br />
BeautifulSoup, Scrapy (for web scraping)<br />

**Datasets:**<br />
The project relies on custom datasets created through manual compilation and web scraping, focusing on key metrics such as population density, traffic patterns, and proximity to key landmarks.

**Modeling:**<br />
Logistic Regression: Baseline model for predicting the presence of metro stops.
<br />Random Forest: Enhanced predictions by capturing complex relationships in the data.
<br />XGBoost: Further refined predictions with gradient boosting for improved accuracy.<br />
Accuracy in Random forest-81%<br />
Accuracy in XGboost-86%<br />
F1 score in Random forest-87%<br />
F1 score in XGboost-91%<br />
<br />**Validation:**<br />
Cross-validation techniques were employed to ensure that the models are both accurate and reliable across different datasets.

![chennai_xg](https://github.com/user-attachments/assets/4d773407-7df7-4ff2-9761-ece5b844dbc8)

![chennai_rf](https://github.com/user-attachments/assets/d600af33-9242-4f6c-8838-e36f82c2fd41)


