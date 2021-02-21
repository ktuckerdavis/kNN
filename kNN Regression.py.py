import streamlit as st
import random
import numpy as np
import scipy.stats
import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import base64

st.title("Concrete Compressive Strength kNN Regression Model")

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('background.png')

st.markdown("""
### Explore the impact of features & k values on regression output.  

###  \t To input query & k values, open the sidebar to the left.

### \t \t To learn more, see below for videos.

""")

my_expander0 = st.beta_expander('README')
with my_expander0:
    st.markdown("""
README for Concrete Compressive Strength kNN Regression Model
 - Current as of 20 Feburary 2021

Modeled and deployed in streamlit as part of EN605.645, Spring 2021, under 
Dr. Stephyn Butcher

**Author**: Kyle Tucker-Davis

**Introduction**: This model seeks to deploy the code from programming assignment 3 
into a UI that allows exploration
of the work done in the module.

**Accompanying files**:

1. ktucke28.py (main file)
2. concrete_compressive_strength.csv (data file)
3. background.png (UI background)

**Sections**:

0. *Background* (see below for reference) changed to picture to show ease of UI use.
1.  *Sidebar*.  Sidebar allows the user to input values for each of the features in 
this module.  Of note, each slider's
min & max & average are that of the training data set.  Default is set to average.
2.  *"Learn about kNN!"* contains links to 3 different YouTube videos to allow users 
to further explore kNN.
It is also included to showcase streamlit functionality for both columns and embedding 
videos.  They are jokingly referenced by length.
3.  *Prediction* returns the query, k value, and model estimation.
4.  *"Interested in regression estimates for other k values"* plots a st.line_chart for 
prediction values using k=1-20.
5.  *"Interested in Correlation/Heatmap?"* shows a heatmap of all features to all the user 
to better understand feature correlation for the 8 distinct features.
6.  *"See the code!"* includes the primary methods and calls used.  It does not include UI 
generation.


**References**:

Note - given eased restrictions on collaboration for this module, several external 
sources were used, as noted below.

 - Background image (before modification): https://wallpapercave.com/wp/ZTRsWNx.jpg

 - Code for using local image as background: 
https://discuss.streamlit.io/t/how-do-i-use-a-background-image-on-streamlit/5067/5

 - Streamlit documentation: https://docs.streamlit.io/en/stable/
 
    """)


my_expander1 = st.beta_expander('Learn about kNN!')
with my_expander1:
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        st.subheader("If you have 5 mins...")
        st.video('https://www.youtube.com/watch?v=HVXime0nQeI')
    with col2:
        st.subheader("If you have 30 mins...")
        st.video('https://www.youtube.com/watch?v=4HKqjENq9OU')
    with col3:
        st.subheader("If you don't have a job...")
        st.video("https://www.youtube.com/watch?v=09mb78oiPkA")

st.sidebar.title("Query Vales")
st.sidebar.subheader("all values default the average")

x0 = st.sidebar.slider("Amount of Cement", 102, 540, 281)
x1 = st.sidebar.slider("Amount of Slag", 0, 360, 74)
x2 = st.sidebar.slider("Amount of Ash", 0, 200, 54)
x3 = st.sidebar.slider("Amount of Water", 122, 247, 181)
x4 = st.sidebar.slider("Amount of Super Platicizer", 0, 32, 6)
x5 = st.sidebar.slider("Amount of Course Aggregate", 801, 993, 972)
x6 = st.sidebar.slider("Amount of Fine Aggregate", 594, 993, 773)
x7 = st.sidebar.slider("Age (Days)", 2, 82, 46)

st.sidebar.subheader("Now! Pick a value for k")
k = st.sidebar.slider("K", 1, 20, 3)

def parse_data(file_name: str):
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = [float(value) for value in line.rstrip().split(",")]
        data.append(datum)
    random.shuffle(data)
    return data

def create_folds(xs, n: int):
    k, m = divmod(len(xs), n)
    # be careful of generators...
    return list(xs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def create_train_test(folds, index: int):
    training = []
    test = []
    for i, fold in enumerate(folds):
        if i == index:
            test = fold
        else:
            training = training + fold
    return training, test

def get_actual(test_fold, num_items):
    actual = []
    [actual.append(test_fold[i][-1]) for i in range(0, num_items)]
    return actual

def sort_and_return(diff_dict, k):
    filtered = sorted(diff_dict.items())
    # print(f'filtered dict: {filtered}')
    total = 0
    for i in range(0, k):
        # print(f'item {i}: {filtered[i]}')
        total += filtered[i][1][-1]
        # print(f'total : {total}')
    return round(total/k, 2)

def evaluation_metric(predicted, actual):
    diff = 0
    for i in range(0, len(predicted)):
        diff += round((predicted[i] - actual[i]) ** 2, 4)
    return round(diff / len(predicted), 4)

def build_knn(database):
    def knn(k, query):
        difference_dict = {}
        for i in range(0, len(database)):
            total = 0
            for j in range(0, len(query)-1):
                diff = round((database[i][j] - query[j])**2, 4)
                total += diff
            difference_dict[round(total / len(query), 4)] = database[i]
        return sort_and_return(difference_dict, k)
    return knn

def knn_model(training_folds, test, k, num_items_to_test):
    predicted = []
    knn = build_knn(training_folds)
    predicted.append(knn(k, test))
    return predicted

def mean_confidence_interval(data, confidence: float=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return round(m, 2), (round(m-h, 2), round(m+h, 2))

data = parse_data("concrete_compressive_strength.csv")
df = pd.DataFrame(np.array(data))
df.columns = ['Cement', 'Slag', 'Ash', 'Water', 'Platicizer', 'CoarseAgg', 'FineAgg', 'Age', 'Strength']

folds = create_folds(data, 10)
train, test = create_train_test(folds, 0)
query = [x0, x1, x2, x3, x4, x5, x6, x7]

st.subheader(f'For query {query} and k={k}')
ans = knn_model(train, query, k, 1)
st.success(f'Estimated strength is {ans[0]}')

my_expander2 = st.beta_expander('Interested in regression estimates for other k values?')
with my_expander2:
    ks = np.arange(1, 21, 1).flatten()
    pred = []
    for item in ks:
        pred.append(knn_model(train, query, item, 1)[0])
    line_data = pd.DataFrame({'k': ks, 'Predicted': pred})
    basic_chart = alt.Chart(line_data).mark_line(point=True).encode(
        x = 'k',
        y = 'Predicted'
    )
    st.altair_chart(basic_chart)

my_expander3 = st.beta_expander('Interested in Correlation/Heatmap?')
with my_expander3:
    fig, ax = plt.subplots()

    sns.heatmap(df.corr(), ax=ax, cmap='RdBu', annot=True)
    st.write(fig)

my_expander4 = st.beta_expander('See the code!')
with my_expander4:
    st.code("""
    
def parse_data(file_name: str):
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = [float(value) for value in line.rstrip().split(",")]
        data.append(datum)
    random.shuffle(data)
    return data

def create_folds(xs, n: int):
    k, m = divmod(len(xs), n)
    # be careful of generators...
    return list(xs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def create_train_test(folds, index: int):
    training = []
    test = []
    for i, fold in enumerate(folds):
        if i == index:
            test = fold
        else:
            training = training + fold
    return training, test

def get_actual(test_fold, num_items):
    actual = []
    [actual.append(test_fold[i][-1]) for i in range(0, num_items)]
    return actual

def sort_and_return(diff_dict, k):
    filtered = sorted(diff_dict.items())
    # print(f'filtered dict: {filtered}')
    total = 0
    for i in range(0, k):
        total += filtered[i][1][-1]
    return round(total/k, 2)

def evaluation_metric(predicted, actual):
    diff = 0
    for i in range(0, len(predicted)):
        diff += round((predicted[i] - actual[i]) ** 2, 4)
    return round(diff / len(predicted), 4)

def build_knn(database):
    def knn(k, query):
        difference_dict = {}
        for i in range(0, len(database)):
            total = 0
            for j in range(0, len(query)-1):
                diff = round((database[i][j] - query[j])**2, 4)
                total += diff
            difference_dict[round(total / len(query), 4)] = database[i]
        return sort_and_return(difference_dict, k)
    return knn

def knn_model(training_folds, test_fold, k, num_items_to_test):
    predicted = []
    knn = build_knn(training_folds)
    [predicted.append(knn(k, test_fold[i])) for i in range(0, num_items_to_test)]
    return predicted

def mean_confidence_interval(data, confidence: float=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return round(m, 2), (round(m-h, 2), round(m+h, 2))

data = parse_data("concrete_compressive_strength.csv") 
folds = create_folds(data, 10)
train, test = create_train_test(folds, 0)
query = [x0, x1, x2, x3, x4, x5, x6, x7]
ans = knn_model(train, query, k, 1)
    """)
