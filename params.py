import numpy as np
from plasticity_rule import Plasticity_Rule as PR
from plasticity_rule import Region
#Fig1
regions_FPLR = [Region('N', (-np.inf, 0.5), 0.5, 0),

           Region('D', (0.5, 1), 0, 0.1),
           Region('P', (1, np.inf), 1, 0.1)]
regions_linear = [Region('N', (-np.inf, 0.5), np.nan, 0),
                  Region('D', (0.5, 1), np.nan, -0.1),
                  Region('P', (1, np.inf), np.nan, 0.2)]

FPLR = PR(regions_FPLR, rule = 'FPLR')
Linear = PR(regions_linear, rule='linear')

#Fig2
hebb_regions = [Region('N', (-np.inf, 0.5), 0, 0),
                  Region('D', (0.5, 0.8), 0, eta),
                  Region('P', (0.8, np.inf), 1, eta)]
anti_hebb_regions = [Region('N', (-np.inf, 0.5), 0, 0),
                     Region('P', (0.5, 0.8), 1, eta),
                     Region('D', (0.8, np.inf), 0, eta)]
titles = ['Fire together wire together',
          'Fire together wire together & \nOut of sync lose your link',
          'Fire together lose your link',
          'Fire together lose your link & \nOut of sync wire together']
all_rules = [PR(hebb_regions), PR(hebb_regions), PR(hebb_regions), PR(anti_hebb_regions)]
all_coeffs = [[0.4,0,0.45,0], [0.55,0,0.7,0], [0.4,0,0.3,0], [0.55,0,0.6,0]]

#Fig3
dicts = [{'alpha': 0.2, 'gamma': 0.2,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.3, 'gamma': 0.3,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.1, 'gamma': 0.6,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.6, 'gamma': 0.1,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.6, 'gamma': 0.6,'theta_D': 0.5, 'theta_P': 1.3},
{'alpha': 0.45, 'gamma': 0.45,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.3, 'gamma': 0.6,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.3, 'gamma': 0.9,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.6, 'gamma': 0.3,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.6, 'gamma': 0.6,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.6, 'gamma': 0.9,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.9, 'gamma': 0.3,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.9, 'gamma': 0.6,'theta_D': 0.5, 'theta_P': 0.8},
{'alpha': 0.9, 'gamma': 0.9,'theta_D': 0.5, 'theta_P': 0.8}]

eta = 1
def rules_from_dict(dicts):
    rules = []
    coeffs = []
    for d in dicts:
        regions = [Region('N', (-np.inf, d['theta_D']), 0.5, 0),
                   Region('D', (d['theta_D'], d['theta_P']), 0, eta),
                   Region('P', (d['theta_P'], np.inf), 1, eta)]
        rules.append(PR(regions))
        coeffs.append([d['alpha'], 0, d['gamma'],0])
    return rules, coeffs

rules, coeffs = rules_from_dict(dicts)