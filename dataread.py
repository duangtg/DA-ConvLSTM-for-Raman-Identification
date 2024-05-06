from binascii import Error
import numpy as np
import pandas as pd


def normalization(data):
    # _range = np.max(data) - np.min(data)
    # return (data - np.min(data)) / _range

    _range = np.max(abs(data))
    return data / _range


def dfbuilder(fin_path, split_df=True, dev_size=0.2, r_state=1, raw=False, mineral_index=0):
    # list of file names with data
    fname_ls = fnamelsbuilder(fin_path)

    # create list to hold dataframes
    df_ls = []
    # read in each file
    if raw:
        df_ls = raw_processing(df_ls, fname_ls, fin_path)

    else:
        for i in fname_ls:
            temp_df = pd.read_csv(fin_path + i, delim_whitespace=False)
            df_ls.append(temp_df)

    # create one large df
    if len(df_ls) > 1:
        df = pd.concat(df_ls)
    else:
        df = df_ls[0]

    if not df[df.isna().any(axis=1)].empty:
        raise Error('the dataframe includes NaN values')

    # if peaks data, additional cleaning
    if 'Peaks Only' in fin_path:
        df = peakscleaning(df)

    # split data for processing
    if split_df:
        return splitdata(df, mineral_index, dev_size, r_state)


def peakscleaning(df):
    """Cleaning for peaks data - drop any rows containing NA

    Args:
        df: a dataframe with peaks data

    Returns:
        DataFrame of peaks data with no NA values
    """
    df.dropna(inplace=True)

    # drop relative intensities
    df.drop(columns=[i for i in df.columns.values if 'val' in i], inplace=True)
    return df


def fnamelsbuilder(fin_path):
    """Build a list of files in directory 'fin_path'.

    Args:
        fin_path: a string providing the path to the folder with the intended files

        In order to avoid unexpected behavior, ensure the fin_path folder only
        contains folders or data files

    Returns:
        A python list of file names in fin_path
    """

    import os
    file_paths = []

    for root, dirs, files in os.walk(fin_path, topdown=True):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    return file_paths


#   #return files
#   return [f for f in listdir(fin_path) if isfile(join(fin_path,f))]
# Process 原始数据


def raw_processing(df_ls, fname_ls, fin_path):
    for fil in fname_ls:
        # temp_df=pd.read_csv(fil,index_col='og-idx',delim_whitespace=False)
        if fil.endswith(".csv"):
            temp_df = pd.read_csv(fil, delim_whitespace=False)
            df_ls.append(temp_df)

    return df_ls


#   #return files
#   return [f for f in listdir(fin_path) if isfile(join(fin_path,f))]

def splitdata(df, mineral_index, dev_size=0.2, r_state=1):
    """
    Splits the dataframe into training and development sets.
    Adds a column indicating the presence of the specified mineral.

    :param df: pandas DataFrame containing the spectra data and mineral compositions.
    :param mineral_index: Index of the mineral column (0 to 4) to be considered.
    :param dev_size: Size of the development set.
    :param r_state: Random state for reproducibility.
    :return: X_train, X_dev, y_train, y_dev, where 'y' contains the specified mineral composition and its presence.
    """

    X = df.copy()
    # separate y from X
    y = X.iloc[:, -4 + mineral_index]
    # Add a binary column indicating the presence of the mineral
    y_binary = (y != 0).astype(int).rename('label_' + str(mineral_index))
    y = pd.concat([y, y_binary], axis=1)

    X.drop(X.columns[-4:], axis=1, inplace=True)

    # split into train and dev sets
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=dev_size, random_state=r_state)
