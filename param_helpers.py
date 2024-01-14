import pandas as pd
import constants
def latex_to_unicode(latex_list):
    # Dictionary mapping LaTeX symbols to Unicode
    latex_to_unicode_dict = {
        r'\theta_{D}': 'θᴅ',
        r'\F_{D}': 'Fᴅ',
        r'\eta_{D}': 'ηᴅ',
        r'\theta_{P}': 'θₚ',
        r'\F_{P}': 'Fₚ',
        r'\eta_{P}': 'ηₚ',
        'alpha': 'α',
        'beta': 'β',
        'gamma': 'γ',
        'delta': 'δ'
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
    df = pd.DataFrame(coeffs, index=['alpha', 'beta', 'gamma', 'delta']).T
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
        row = {'rule': rule_name}
        row.update(rule.to_pandas().iloc[0].to_dict())
        if coeffs is not None:
            print(coeffs_to_pandas(coeff))
            row.update(coeffs_to_pandas(coeff).iloc[0].to_dict())
        row_df = pd.DataFrame(row, index=[0])  # Create a DataFrame from the row
        mydf = pd.concat([mydf, row_df], ignore_index=True)  # Use concat instead of append

    print(mydf)
    mydf.to_csv(constants.PARAMS_FOLDER + f'fig{fig_num}_params.csv', index=False)


print(coeffs_to_pandas([0.3,0,0.3,0]))