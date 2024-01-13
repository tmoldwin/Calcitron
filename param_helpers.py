def latex_to_unicode(latex_list):
    # Dictionary mapping LaTeX symbols to Unicode
    latex_to_unicode_dict = {
        r'\theta_{D}': 'θᴅ',
        r'\F_{D}': 'Fᴅ',
        r'\eta_{D}': 'ηᴅ',
        r'\theta_{P}': 'θₚ',
        r'\F_{P}': 'Fₚ',
        r'\eta_{P}': 'ηₚ',
        r'\alpha': 'α',
        r'\beta': 'β',
        r'\gamma': 'γ',
        r'\delta': 'δ'
    }

    # Replace LaTeX symbols with Unicode in each string in the list
    unicode_list = []
    for latex_str in latex_list:
        for latex, unicode in latex_to_unicode_dict.items():
            latex_str = latex_str.replace(latex, unicode)
        unicode_list.append(latex_str)

    return unicode_list

# Test the function
latex_list = [r'$\theta_{D}$',  r'$\F_{D}$',  r'$\eta_{D}$',  r'$\theta_{P}$',  r'$\F_{P}$',  r'$\eta_{P}$', r'\alpha', r'\beta', r'\gamma', r'\delta']
unicode_list = latex_to_unicode(latex_list)
print(unicode_list)