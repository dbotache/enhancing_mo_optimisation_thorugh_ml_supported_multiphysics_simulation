import os
import pandas as pd
import ipywidgets as widgets
import msoffcrypto
import io
import json


def load_df(args):

    X_train, X_test = pd.read_hdf(f'{args.main_path}/Dataframes/{args.splitfolder}/X_train_{args.file_name}.h5'), \
                      pd.read_hdf(f'{args.main_path}/Dataframes/{args.splitfolder}/X_test_{args.file_name}.h5')
    y_train, y_test = pd.read_hdf(f'{args.main_path}/Dataframes/{args.splitfolder}/y_train_{args.file_name}.h5'), \
                      pd.read_hdf(f'{args.main_path}/Dataframes/{args.splitfolder}/y_test_{args.file_name}.h5')

    return X_train, X_test, y_train, y_test


def get_file_dropdown(path, sub_folder=None):
    
    if sub_folder is None:
        data_path = path
        str_filter = 'X_'
    else:
        data_path = os.path.join(path, sub_folder)
        str_filter = 'X_train_'
    
    files_available = [i for i in os.listdir(data_path) if str_filter in i]
    files_available = [i.replace(str_filter, "") for i in files_available]
    files_available = [i.replace(".h5", "") for i in files_available]
    
    file_dropdown = widgets.Dropdown(
        options=files_available,
        value=files_available[0],
        description='Data name:',
        disabled=False,
    )

    return file_dropdown


def load_df_from_xlsx(data_path, file_name, pw=None):
    X = pd.DataFrame()
    y = pd.DataFrame()

    config_data_path = os.path.join(data_path, file_name.replace('.xlsx', '.json'))

    with open(config_data_path, 'r') as f:
        config = json.load(f)
        input_size = config['dimensions']['input_size']

    if pw is None:
        df= pd.read_excel(os.path.join(data_path, file_name),
                          sheet_name='database',
                          header=None)
        X, y = split_features_targets(df, input_size)

    else:

        try:
            df = return_df_from_encr_xlsx(data_path, file_name, pw)
            X, y = split_features_targets(df, input_size)
            print(f'Input size: {input_size}, output size: {y.shape[1]}')
        except:
            print('wrong password')

    return X, y


def split_features_targets(df, input_size=29):
    df = pd.DataFrame(df.transpose().values[1:],
                      index=None,
                      columns=df.transpose()[0:1].values.flatten(),
                      dtype=float)

    feature_names = df.columns.values[0:input_size]
    target_names = df.columns.values[input_size:]

    X = df.loc[:, feature_names]
    y = df.loc[:, target_names]

    return X, y


def return_df_from_encr_xlsx(data_path, file_name, pw):
    decrypted_workbook = io.BytesIO()
    file_path = os.path.join(data_path, file_name)

    with open(file_path, 'rb') as file:
        office_file = msoffcrypto.OfficeFile(file)
        office_file.load_key(password=pw)
        office_file.decrypt(decrypted_workbook)
        df = pd.read_excel(decrypted_workbook, header=None)
    return df


def load_ui(data_path):
    exclude = ['.ipynb_checkpoints', '.idea']
    files_available = [i for i in os.listdir(data_path) if i not in exclude and '.xlsx' in i]

    file_name = widgets.Dropdown(
        options=files_available,
        value=None,
        description='Data name:',
        disabled=False,
    )

    pw = widgets.Password(
        value=None,
        placeholder='Enter password',
        description='Password:',
        disabled=False
    )

    ui = widgets.HBox([file_name, pw])

    def f(file_name_, pw_):
        if file_name_ is not None and pw_ is not None:
            print(file_name_, ';', pw_)

    out = widgets.interactive_output(f, {'file_name_': file_name, 'pw_': pw})
    display(ui)

    return out


def load_data_with_ui(data_path, out):
    X = pd.DataFrame()
    y = pd.DataFrame()

    if out.outputs != ():
        file_name = out.outputs[0]['text'].split(';')[0].replace(' ', '')
        pw = out.outputs[0]['text'].split(';')[1].replace(' ', '')[:-1]

        if pw == '':
            pw = None

        return load_df_from_xlsx(data_path, file_name, pw)

    return X, y