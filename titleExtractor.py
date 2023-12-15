import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np
from wordcloud import WordCloud
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import shap

'''a class to extract titles by using lightGBM'''

class titleExtractor:
    def __init__(self):
        self.df_train = None  # to keep train data
        self.df_test = None  # to keep test data
        self.param_grid = {
            'n_estimators': [200],
            'learning_rate': [0.1, 0.2],
            'max_depth': [4, 5],
            'num_leaves': [40, 50],
            'min_data_in_leaf': [500, 600],
        }  # lightGBM parameter search

        # sentence embeddings
        self.st_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def read_csv_to_df(self, file_path_train: str, file_path_test: str) -> None:
        """
        Read csv files
        :param file_path_train: train data
        :param file_path_test: test data
        :return: set class attribute train_df and test_df
        """

        columns = ['Text', 'IsBold', 'IsItalic', 'IsUnderlined', 'Left', 'Right', 'Top',
                   'Bottom', 'FontType', 'Label']

        try:
            self.df_train = pd.read_csv(file_path_train, encoding='ISO-8859-1')
            self.df_train = self.df_train[columns]
            print(f"DataFrame loaded for {file_path_train} from {file_path_train}")
        except Exception as e:
            print(f"Error loading DataFrame for {file_path_train}: {str(e)}")

        try:
            self.df_test = pd.read_csv(file_path_test, encoding='ISO-8859-1')
            self.df_test = self.df_test[columns]
            print(f"DataFrame loaded for {file_path_test} from {file_path_test}")
        except Exception as e:
            print(f"Error loading DataFrame for {file_path_test}: {str(e)}")

    def print_data_information(self, df: pd.DataFrame) -> None:
        """
        Print some important stats such as mean, missing values, etc .
        :param df: train or test df to be checked
        :return: None
        """
        print(f'Total number of instances in train data {len(df)}')
        print('Missing values\n')
        print(df.isna().sum())
        print(f'Name of columns {df.columns}')
        print(df.info())
        print('-----------------')
        print('Printing describe: \n', df.describe())
        print('-----------------')
        print('Printing data types: \n', df.dtypes)

    def feature_normalization(self, df: pd.DataFrame) -> DataFrame:
        """
        Feature normalization
        :param df: test or train df
        :return: updated df
        """

        numerical_columns = ['Left', 'Right', 'Top', 'Bottom', 'text_length', 'width', 'height',
                             'center_x', 'center_y', 'area', 'aspect_ratio']

        # Normalize features
        scaler = StandardScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

        return df

    @staticmethod
    def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert boolean values to int for easier handling
        :return: updated df
        """

        df[df.select_dtypes(include='bool').columns] = df.select_dtypes(include='bool').astype(int)
        return df

    def visualize_corr_map(self):
        """
        Calculate corr map on train data
        :param:
        :return:
        """

        numeric_columns = self.df_train.select_dtypes(include='number').columns.difference(['Label', 'Text'])

        # Calculate the correlation matrix for numeric columns
        corr_train = self.df_train[numeric_columns].corr()

        sns.set(font_scale=1.1)
        mask = np.triu(corr_train.corr())
        plt.figure(figsize=(20, 20))
        sns.heatmap(corr_train, annot=True, fmt='.1f', cmap='coolwarm', square=True, mask=mask, linewidth=1, cbar=True)
        plt.savefig('../output/correlation_heatmap.png')
        plt.show()

    def visualize_barplot(self):
        """
        Plot bar graphs for numerical features
        :return:
        """
        def barplot(feature):
            self.df_train[feature].hist()
            plt.title('Count')
            plt.xlabel(feature)
            plt.ylabel('Count')
            figure_name = '../output/' + feature + '_barplot.png'
            plt.savefig(figure_name)
            plt.show()

        for column_name in self.df_train.select_dtypes(include='int64').columns:
            barplot(column_name)

    def visualize_plot_groupby(self, df):
        """
        Group by label and see how features differ accordingly
        :param df: data
        :return:
        """
        def plot_groupby(column_name):
            count_combinations = df.groupby([column_name, 'Label']).size().unstack(fill_value=0)
            count_combinations.plot(kind='bar', stacked=True, color=['red', 'blue'])
            plt.xlabel('IsBold')
            plt.ylabel('Count')
            ptitle = 'Counts of ' + column_name + ' and Label Combinations'
            plt.title(ptitle)
            plt.legend(title='Label', loc='upper right')
            figure_name = '../output/' + column_name + '_groupby.png'
            plt.savefig(figure_name)
            plt.show()

        plot_groupby('IsBold')
        plot_groupby('IsUnderlined')
        plot_groupby('IsItalic')

    def plot_distribution(self, df):
        '''
        Plot distrubution of word lengths for title and non-title
        :param df:
        :return:
        '''
        title_word_len = df[df['Label'] == 1]['Text'].str.split().map(lambda x: len(x))
        non_title_word_len = df[df['Label'] == 0]['Text'].str.split().map(lambda x: len(x))

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Title", "Non-title"))

        fig.add_trace(
            go.Histogram(x=title_word_len, marker_line=dict(color='black'), marker_line_width=1.2),
            row=1, col=1
        ).add_trace(
            go.Histogram(x=non_title_word_len, marker_line=dict(color='black'), marker_line_width=1.2),
            row=1, col=2
        ).update_layout(title_text="Length of words", title_x=0.5, showlegend=False).show()

        fig.write_html("../output/word_len_distribution.html")

    def plot_avg_word(self, df):
        """
        Plot avg word len
        :param df:
        :return:
        """
        def avgwordlen(strlist):
            sum = []
            for i in strlist:
                sum.append(len(i))
            return sum

        avg_word_len1 = df[df['Label'] == 1]['Text'].str.split().apply(avgwordlen).map(lambda x: np.mean(x))
        avg_word_len2 = df[df['Label'] == 0]['Text'].str.split().apply(avgwordlen).map(lambda x: np.mean(x))

        group_labels = ['Title', 'Non title']
        colors = ['rgb(0, 0, 100)', 'rgb(0, 200, 200)']

        fig = ff.create_distplot([avg_word_len1, avg_word_len2], group_labels, bin_size=.2, colors=colors, )

        fig.update_layout(title_text="Average word length", title_x=0.5, xaxis_title="Text",
                          yaxis_title="Density").show()

        fig.write_html("../output/avg_word_len_bar.html")

    def visualize_word_cloud(self, df):
        """
        Plot word cloud
        :param df:
        :return:
        """
        text = ' '.join(df[df['Label'] == 1]['Text'].astype(str))
        wordcloud = WordCloud(width=800, height=400, max_words=200, background_color='white').generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud for Title')
        plt.show()

        text = ' '.join(df[df['Label'] == 0]['Text'].astype(str))
        wordcloud = WordCloud(width=800, height=400, max_words=200, background_color='white').generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud for Non-Title')
        plt.savefig('../output/word_cloud.png')
        plt.show()

    def visualize_boxplot(self, df):
        """
        visualize boxplot
        :param df:
        :return:
        """
        df.boxplot()
        plt.show()

        sns.countplot(x='Label', data=df)
        plt.show()

    def visualize_histplot(self, df):
        """
        Plot histplot
        :param df:
        :return:
        """
        sns.histplot(df['text_length'], kde=True, color='orange', bins=10)
        plt.xlabel('Text Length')
        plt.ylabel('Density')
        plt.title('Distribution of Text Lengths')
        plt.savefig('../output/histplot.png')
        plt.show()

    def _percentage_upper_letters(self, text: str) -> float:
        """
        Add a feature column which is percentage of upper letters
        :param text: data
        :return: float
        """
        total_chars = len(text)
        upper_chars = sum(1 for char in text if char.isupper())
        return (upper_chars / total_chars) * 100 if total_chars > 0 else 0

    def _percentage_words_starting_upper(self, text: str) -> float:
        """
        Add a feature column percentage of words starting with upper case
        :param text: text data
        :return: float
        """
        words = text.split()
        total_words = len(words)
        upper_start_words = sum(1 for word in words if word and word[0].isupper())
        return (upper_start_words / total_words) * 100 if total_words > 0 else 0

    def add_new_feats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add new features to df
        :param df: data
        :return: updated df
        """

        df['Text'].fillna('', inplace=True)
        df['Text'] = df['Text'].astype(str)

        df['percent_alpha_numeric'] = df['Text'].apply(
            lambda x: (sum(c.isalnum() for c in x) / len(x) * 100) if len(x) > 0 else 0)
        df['percentage_upper_letters'] = df['Text'].apply(self._percentage_upper_letters)
        df['percentage_words_starting_upper'] = df['Text'].apply(self._percentage_words_starting_upper)
        df['text_length'] = df['Text'].apply(len)
        df['width'] = df['Right'] - df['Left']
        df['height'] = df['Bottom'] - df['Top']
        df['center_x'] = (df['Left'] + df['Right']) / 2
        df['center_y'] = (df['Top'] + df['Bottom']) / 2
        df['area'] = df['width'] * df['height']
        df['aspect_ratio'] = df['width'] / df['height']

        return df

    def explain_shap(self, best_model, X_train):
        """
        Use SHAP to explain feature importance
        :param best_model: best lightGBM model
        :param X_train: train data
        :return: None, only visualize
        """
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_train)
        shap.summary_plot(shap_values, X_train)

        plt.savefig('../output/shap_summary_plot.png')
        plt.close()

    def predict_and_test(self, best_model, best_params, x_test, y_test):
        """
        Predict a given test set given a model and print metrics
        :param best_model:
        :param best_params:
        :param x_test:
        :param y_test:
        :return:
        """
        y_pred = best_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        print("Best Parameters:", best_params)
        print("Test Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score:", f1)
        print("Confusion Matrix:\n", conf_matrix)

    def classification(self):
        """
        Classify data into title and non-title using lightGBM
        :return:
        """
        X = self.df_train.drop(['Label', 'Text'], axis=1)
        y = self.df_train['Label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2018)

        clf = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', is_unbalance=True)
        search = RandomizedSearchCV(clf, param_distributions=self.param_grid, n_iter=5, scoring='accuracy',
                                    random_state=2018)
        search.fit(X_train, y_train)

        best_params = search.best_params_
        best_model = search.best_estimator_

        self.predict_and_test(best_model, best_params, X_test, y_test)
        self.explain_shap(best_model, X_train)

        x_test = self.df_test.drop(['Label', 'Text'], axis=1)
        y_test = self.df_test['Label']

        self.predict_and_test(best_model, best_params, x_test, y_test)

    def extract_embeddings_features(self, df):
        """
        Add embeddings feature
        :param df: data
        :return: updated df
        """

        df['embeddings'] = df['Text'].apply(lambda x: self.st_model.encode([x[:1024]])[0])

        vector_column = df['embeddings']
        vector_df = pd.DataFrame(list(vector_column),
                                 columns=[f'Feature_{i + 1}' for i in range(vector_column.iloc[0].shape[0])])
        df = pd.concat([df, vector_df], axis=1)

        df = df.drop(['embeddings'], axis=1)

        return df
