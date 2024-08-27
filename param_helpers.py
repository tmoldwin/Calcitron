import pandas as pd
import constants
def latex_to_unicode(latex_list):
    # Dictionary mapping LaTeX symbols to Unicode
    latex_to_unicode_dict = {
        r'\theta_{D}': 'θ_D',
        r'\F_{D}': 'F_D',
        r'\eta_{D}': 'η_D',
        r'\theta_{P}': 'θ_P',
        r'\F_{P}': 'F_P',
        r'\eta_{P}': 'η_P',
        r'\theta_{PPNZ}': 'η_PPNZ',
        r'\F_{PPNZ}': 'F_PPNZ',
        r'\eta_{PPNZ}': 'η_PPNZ',
    }

    # Replace LaTeX symbols with Unicode in each string in the list
    unicode_list = []
    for latex_str in latex_list:
        for latex, unicode in latex_to_unicode_dict.items():
            latex_str = latex_str.replace(latex, unicode)
            latex_str = latex_str.replace('$', '')
        unicode_list.append(latex_str)

    return unicode_list

def coeffs_to_pandas(coeffs):
    df = pd.DataFrame(coeffs, index=['α', 'β', 'γ', 'δ']).T
    df.columns = latex_to_unicode(df.columns)
    return df


def fig_params(rules, rule_names, fig_num, coeffs=None):
    mydf = pd.DataFrame()
    # Check if coeffs is None
    if coeffs is None:
        zipped = zip(rules, rule_names)
    else:
        zipped = zip(rules, rule_names, coeffs)
    print(zipped.__str__())
    for values in zipped:
        if coeffs is None:
            rule, rule_name = values
        else:
            rule, rule_name, coeff = values
        row = {'Panel': rule_name}
        row.update(rule.to_pandas().iloc[0].to_dict())
        if coeffs is not None:
            print(coeffs_to_pandas(coeff))
            row.update(coeffs_to_pandas(coeff).iloc[0].to_dict())
        row_df = pd.DataFrame(row, index=[0])  # Create a DataFrame from the row
        mydf = pd.concat([mydf, row_df], ignore_index=True)  # Use concat instead of append
    mydf.columns = latex_to_unicode(mydf.columns)
    mydf = mydf.replace('threshold', 'step')
    mydf.to_csv(constants.PARAMS_FOLDER + f'fig{fig_num}_params.csv', index=False, encoding='utf-8-sig')



def param_concat():
    df = pd.DataFrame()
    # Initialize an empty list to store the DataFrames
    dfs = []

    # Loop over the file numbers
    for i in range(1, 9):  # Replace n with the number of files
        # Read the CSV file into a DataFrame
        if not i in [3]:
            df_temp = pd.read_csv(constants.PARAMS_FOLDER + f'fig{i}_params.csv')

            # Create a DataFrame with a single row containing the figure number
            df_fig = pd.DataFrame({'Fig': [f'Fig{i}']})

            # Append the figure number DataFrame and the CSV DataFrame to the list
            dfs.append(df_fig)
            dfs.append(df_temp)

            # Concatenate all the DataFrames in the list vertically
            df = pd.concat(dfs, ignore_index=True)

            # Print the resulting DataFrame
            print(df)
    df.to_csv(constants.PARAMS_FOLDER + f'all_params.csv', index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    param_concat()
