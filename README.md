# DECISION-TREE-IMPLEMENTATION

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: P VISHNU SIDDARTH

*INTERN ID*: CT04DF2078

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

##📌 Project Overview
This project is developed as part of the internship task provided by CODTECH, with the objective to build, train, and visualize a Decision Tree Classifier using the Scikit-learn library. The model is applied to a classic dataset — the Iris flower dataset, which is widely used for classification algorithm testing and performance evaluation. The goal is to classify iris flowers into one of three species based on their petal and sepal dimensions.

The project focuses on machine learning fundamentals, including:

Data loading and preprocessing

Model training using Decision Tree

Evaluation of accuracy and classification performance

Visualization of the decision tree structure

Confusion matrix for performance analysis

By the end of this project, we successfully built a clean, interpretable model with clear visual output — which is essential in real-world decision-making scenarios where model transparency is crucial.

🧠 Dataset Description
The Iris dataset is a multivariate dataset introduced by Ronald A. Fisher. It consists of 150 samples, each belonging to one of three classes: Iris setosa, Iris versicolor, and Iris virginica. Each sample includes four numerical features:

Sepal length (cm)

Sepal width (cm)

Petal length (cm)

Petal width (cm)

The dataset is well-balanced with 50 samples per class and is available directly through sklearn.datasets.

⚙️ Technology & Libraries Used
Python – Main programming language

Scikit-learn – For model building, training, and evaluation

Pandas – For data handling

Matplotlib & Seaborn – For data and result visualization

Jupyter Notebook – For structured code and markdown documentation

🚀 Steps Performed
1. Data Loading
The dataset was loaded using load_iris() from sklearn.datasets and stored in a pandas DataFrame for easy manipulation.

2. Data Splitting
The dataset was split into training (70%) and testing (30%) sets using train_test_split() to ensure unbiased evaluation.

3. Model Building
A DecisionTreeClassifier was initialized with:

criterion='entropy' for information gain

max_depth=3 to avoid overfitting and enhance visual clarity

The model was trained on the training data using .fit().

4. Prediction & Evaluation
Predictions were made using .predict() on the test set. The model’s performance was measured using:

Accuracy score

Classification report (precision, recall, F1-score)

Confusion matrix

5. Visualization
The decision tree was visualized using plot_tree() with options:

filled=True

rounded=True
This allows each node to show class distribution, entropy, and decision rules.

Additionally, a heatmap of the confusion matrix was generated using Seaborn, which provides insight into how many predictions were correct or incorrect per class.

📊 Results
The model achieved high accuracy with clean separation between the three iris species. Key highlights:

All setosa samples were classified correctly (perfect separation in early nodes)

Minor misclassifications between versicolor and virginica, as expected due to overlapping feature values

Visual tree shows clear, interpretable decision paths

🎯 Learning Outcomes
Understood how decision trees split data using entropy (information gain)

Learned to visualize ML models to interpret their decisions

Gained hands-on experience with the end-to-end model development process

Explored performance evaluation using standard metrics and confusion matrices

📁 Deliverables
A complete Jupyter Notebook containing:

Code

Tree visualizations

Confusion matrix

Accuracy & report

Screenshots of output tree and evaluation plots

Well-documented analysis and markdown cells explaining each step

🏁 Internship Completion
This project serves as the final submission for the Decision Tree implementation module of the CODTECH internship program. Upon completion and review of this notebook, a certificate of internship completion will be issued.

🔗 Conclusion
This project demonstrates the practical implementation of a machine learning classification algorithm using a structured approach. The decision tree model not only classifies data effectively but also provides transparency in how decisions are made. This is a critical step in learning how machine learning can be applied in real-world classification tasks across industries such as healthcare, finance, and marketing.

