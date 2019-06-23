import pandas as pd

population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
# print(population)


area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)

# make data fram usinb this two series of objects

states = pd.DataFrame({'population': population, 'Area': area})

states['density'] = states['population'] / states['Area']


print(states.loc[states.density < 100, ['population', 'Area', 'density']])
