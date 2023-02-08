import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error, mean_absolute_error
import os
import json
import argparse

import matplotlib as mpl

plt.rcParams['axes.facecolor']='white'

mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use('seaborn-colorblind')

mpl.rcParams['axes.spines.top']=False
mpl.rcParams['axes.spines.right']=False
colors={1:'#a1a3e7', 2:'#878689', 3:'#f29ed0'}



# Instantiate the parser
parser = argparse.ArgumentParser(description='')

# Required positional argument
parser.add_argument('data', type=str,
                    help='A path to .xlsx data file')
# Optional positional argument
parser.add_argument('--output', type=str,
                    help='A path to write results. Default is ./results')

maxes={'Болевой фактор7':20,
 'Психологический фактор7':45,
 'Физический фактор7':20,
 'Социальный фактор7':15,
 'ВАШ7':10,
      'Болевой':20,
 'Психологический':45,
 'Физический':20,
 'Социальный':15,
 'ВАШ':10}
mins={'Болевой фактор7':4,
 'Психологический фактор7':9,
 'Физический фактор7':4,
 'Социальный фактор7':3,
 'ВАШ7':0,
      'Болевой':4,
 'Психологический':9,
 'Физический':4,
 'Социальный':3,
 'ВАШ':0}

treatment={'group1':'фармакотерапия',
           'group2':'кроссэктомия',
           'group3':'флебэктомия'}


def train_and_predict(targs, data, results, groups, train=False, path=''):
    rf = sklearn.ensemble.RandomForestRegressor()
    ind=data.index
    if train:
        ind_train, ind_test = train_test_split(ind, test_size=0.2, random_state=42)

    res_dict = {}
    for tar in targs:

        res_dict[tar + '1'] = {}
        res_dict[tar + '2'] = {}

        if train:
            pc = results[tar]['features']
            X = data[pc + list(groups)].astype('float')
            X[pc] = X[pc] - X[pc].min().min()
            X[pc] = X[pc] / X[pc].max().max()
            Y = data[tar]
            rf = rf.set_params(**results[tar]['best_params_'])
            model_fitted = rf.fit(X.loc[ind_train], Y.loc[ind_train])
            r2 = model_fitted.score(X.loc[ind_train], Y.loc[ind_train])
            results[tar]['r2_test'] = r2
            results[tar]['MAE_test'] = mean_absolute_error(model_fitted.predict(X.loc[ind_test]), Y.loc[ind_test])
            model_fitted = rf.fit(X, Y)
            joblib.dump(model_fitted, tar + '.joblib')
            with open('results_test.json', 'w', encoding='utf-8') as f:
                json.dump(results, f)
        else:
            r2 = results[tar]['r2']
            model_fitted = joblib.load(path + tar + '.joblib')

        pc = results[tar]['features']

        for gr in groups:
            pc0 = [f + gr if 'predicted' in f else f for f in pc]
            X = data[pc0 + list(groups)].astype('float')
            X[pc0] = X[pc0] - X[pc0].min().min()
            X[pc0] = X[pc0] / X[pc0].max().max()
            X[gr] = [1] * X.shape[0]
            X[groups[groups != gr]] = [[0, 0]] * X.shape[0]
            # print(X)
            data[tar + '_predicted' + gr] = model_fitted.predict(X)
            factor = tar.split(' ')[0]
            res_dict[tar + '1'][gr] = (model_fitted.predict(X) - mins[factor]) / (maxes[factor] - mins[factor]) * r2
            if 'ВАШ' not in factor:
                nullday = factor + ' фактор 0'
            else:
                nullday = factor + ' 0'
            res_dict[tar + '2'][gr] = (data[nullday] - mins[factor]) / (maxes[factor] - mins[factor]) * r2 - (
                        model_fitted.predict(X) - mins[factor]) / (maxes[factor] - mins[factor]) * r2


    return res_dict, data

def plots_per_person(data, results, days, factors, groups, output):
    for per in data.index:
        fig, ax = plt.subplots(1, 5, figsize=(16, 6))
        for i, f in enumerate(factors):
            f_arr = []
            gr_arr = []
            for j, gr in enumerate(groups):
                for d in days[1:]:
                    f_arr = f_arr + [data.loc[per, f + d + '_predicted' + gr]]
                    gr_arr = gr_arr + [treatment[gr]]
            df0 = pd.DataFrame({f: f_arr,
                                'day': np.ravel([days[1:]] * len(groups)),
                                'Лечение': gr_arr})
            sns.pointplot(
                # data=data_pd.astype('int'), x="day", y=f, hue="Group",
                data=df0,
                x="day", y=f, hue='Лечение',
                capsize=.1,
                palette=colors.values(),
                dodge=True,
                estimator=np.median,
                # errorbar=('ci', 50),
                kind="point", ax=ax[i]
            )
            mae = [results[f + d]['MAE_test'] for d in days[1:]]
            for j, gr in enumerate(treatment.values()):
                df00 = df0[df0['Лечение'] == gr]
                ax[i].fill_between(df00.day, df00[f] + mae, df00[f] - mae,
                                   alpha=0.2, color=colors[j + 1])
            ax[i].set_title(f)

            ax[i].set_xlabel('день')
            ax[i].set_ylabel('')
            if i != 4: ax[i].get_legend().remove()
        fig.patch.set_facecolor('white')
        plt.savefig(output + str(per) + 'prediction.png', dpi=300)
        plt.close()

def main():
    args = parser.parse_args()

    datafile = args.data
    output = args.output
    if output==None:
        output='results/'
    if not os.path.exists(output):
        print('Path '+output+' not exists. Creating directory.')
        os.mkdir(output)

    data=pd.read_excel(datafile)
    data['age'] = data['возраст']

    days = ['0', '7', '14', '28', '45']
    factors = ['Болевой фактор ', 'Психологический фактор ', 'Физический фактор ', 'Социальный фактор ', 'ВАШ ']
    data['Сопутствующие заболевания bin'] = data['Сопутствующие заболевания (1 или 0)']
    data = data.replace(
        {'М': 0, 'Ж': 1, 'Острый': 2, 'Стихающий': 1, 'Стихший': 0, 'Стихший ': 0, 'Острый ': 2, 'Стихающий ': 1})
    data.columns = [c.strip() for c in data.columns]

    groups = np.array(['group1', 'group2', 'group3'])
    data[groups]=[[np.nan]*3]*data.shape[0]
    with open('models/info.json', 'r') as f:
        results = json.load(f)
    targs = list(results.keys())
    res_dict, data_predicted = train_and_predict(targs, data, results, groups, train=False,
                                              path="models/",
                                              )
    plots_per_person(data_predicted, results, days, factors, groups, output)

    res_pd = pd.DataFrame()
    for f in list(res_dict.keys()):
        if f[-1] == '2': continue
        weight = 1
        df = pd.DataFrame(res_dict[f], index=data.index)
        if res_pd.shape[0] > 0:
            res_pd = res_pd + weight * df
        else:
            res_pd = weight * df
    res_pdm = res_pd.idxmin(axis=1)
    res_pdm = res_pdm.apply(lambda x: x[-1:]).astype('int')
    data_predicted.columns=[c if 'predicted' not in c else c[:-6]+'_'+treatment[c[-6:]] for c in data_predicted.columns]
    data_predicted['predicted treatment'] = res_pdm.replace({1:'фармакотерапия',
           2:'кроссэктомия',
           3:'флебэктомия'})

    data_predicted.to_excel(output+'prediction.xlsx')




if __name__ == '__main__':
    main()