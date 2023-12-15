import logging
from titleExtractor.src.titleExtractor import titleExtractor
import argparse

logging.basicConfig(format='%(asctime)s %(levelname)-8s  %(filename)-2s :: %(funcName)-2s :: %(message)s',
                    level=logging.INFO)
def set_args():
    """
    Set arguments for sentiment classifier
    :return: args object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--classify', help='Perform classification')
    parser.add_argument('--visualisation', help='Perform visualisation')
    args = parser.parse_args()
    return args


def handle_columns_data(te):
    '''
    For train and test data, convert data types
    Drop FontType as it has only one value
    :param te:
    :return:
    '''
    te.df_train = te.convert_data_types(te.df_train)
    te.df_test = te.convert_data_types(te.df_test)

    te.df_train = te.df_train.drop(['FontType'], axis=1)
    te.df_test = te.df_test.drop(['FontType'], axis=1)

def exploratory_data_analysis(te):
    '''
    Do visualizations to do exploratory data analysis
    :param te:
    :return:
    '''
    te.visualize_barplot()
    te.visualize_histplot(te.df_train)
    te.visualize_plot_groupby(te.df_train)
    te.plot_distribution(te.df_train)
    te.plot_avg_word()
    te.visualize_word_cloud()

if __name__ == '__main__':


    te = titleExtractor()

    logging.INFO('reading data ...')
    te.read_csv_to_df('../data/train_sections_data.csv', '../data/test_sections_data.csv')

    logging.INFO('\nPrinting TRAIN data information ********************** \n')
    # te.print_data_information(te.df_train)

    logging.INFO('\nPrinting TEST data information ********************** \n ')
    # te.print_data_information(te.df_test)

    logging.INFO('\nConvert data types ********************** \n ')
    handle_columns_data(te)

    logging.INFO('\nAdding new features ********************** \n ')
    te.df_train = te.add_new_feats(te.df_train)
    te.df_test = te.add_new_feats(te.df_train)

    # logging.INFO('\nExploratory analysis ********************** \n ')
    # exploratory_data_analysis(te)

    logging.INFO('\nCorrelation map ********************** \n ')
    te.visualize_corr_map()

    logging.INFO('\nFeature normalization ********************** \n ')
    te.df_train = te.feature_normalization(te.df_train)
    te.df_test = te.feature_normalization(te.df_test)#

    logging.INFO('\nClassification ********************** \n ')
    te.classification()

    logging.INFO('\nAdding embedding feats ********************** \n ')
    te.df_train = te.extract_embeddings_features(te.df_train)
    te.df_test = te.extract_embeddings_features(te.df_test)

    logging.INFO('\nClassifying again ********************** \n ')
    te.classification()





