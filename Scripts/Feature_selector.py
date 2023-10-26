from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import shap

from Trained_model import Trained_model
import numpy as np
import pandas as pd
import warnings
import pickle
warnings.filterwarnings('ignore')
from IPython.display import clear_output
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import make_scorer

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly

def func_mcc(y_true, y_pred):
    score = mcc(y_true, y_pred)
    return score
    
def func(y_true, y_pred):
    score = mcc(y_true, y_pred)
    return score

def mcc_scorer():
    return make_scorer(func)

metrics = {'roc_auc' : 'roc_auc', 'mcc' : mcc_scorer()}

class Feature_selector(Trained_model):
    '''
    Class for feature selection.
    Traget and dataset should be specified.
    Tuned parameters of model are read, and model is retrained.
    Chosen selectors are used to determine top N features. 
    '''
    def __init__(self, targets, datasets, subsets, models, selectors, folder, drop_col):
        super().__init__(targets, datasets, subsets, models, selectors,)        
        self.folder = folder
        self.drop_col = drop_col
      
    def get_shap_values_summary(self, model_instance, X_train, X_test, model_name, path,):
        '''
        Function calculates shap values.
        Calculation is different for tree-based models and not-tree-based models.
        Bar and violin plots are made.
        '''

        # TODO: fix SHAP values calculation for not-tree-based models

        if model_name == 'svm' or model_name == 'LogisticRegression' or model_name == 'adaboost':
            # returns zeros and I dont know why
            explainer = shap.Explainer(model_instance.predict, pd.concat([X_train, X_test]))
            shap_values = explainer(pd.concat([X_train, X_test]))
            values = shap_values.values[:,:]

        if model_name == 'randomforest' or model_name == 'catboost':
            explainer = shap.TreeExplainer(model=model_instance, 
                                          model_output='raw', 
                                          feature_perturbation='interventional',)
            shap_values = explainer(pd.concat([X_train, X_test]).values)
            values = shap_values.values[:,:,0]
        
        # with open(f'{path}{model_name}_shap_values.pickle', 'wb') as f:
        #     pickle.dump(shap_values, f)
        # pd.DataFrame(shap_values[0]).to_excel(f'{path}{model_name}_shap_values.xlsx')
        # # save values
        f = pd.DataFrame(np.stack([X_train.columns, abs(values).mean(axis=0)], axis=1), columns=['features', 'values'])
        if self.save:
            f.to_excel(f'{path}{model_name}_shap_values.xlsx')
        f = None

        # plot graph
        if True:
            # get Bar and Violin plots
            shap.summary_plot(shap_values=values, features=pd.concat([X_train, X_test]), feature_names=X_train.columns, plot_type="bar", plot_size=(19.5,6.75), show=False)
            plt.title(label='SHAP values', loc='center', pad=None, y=0.98,
                    fontdict={'fontsize': 16,
                                # 'fontweight' : rcParams['axes.titleweight'],
                                'verticalalignment': 'baseline',
                                'horizontalalignment': 'center'})
            if self.save:
                plt.savefig(f'{path}{model_name}_barplot.pdf')
            # plt.show()
            plt.clf()
            shap.summary_plot(shap_values=values, features=pd.concat([X_train, X_test]), feature_names=X_train.columns, plot_type="layered_violin", color='coolwarm', plot_size=(19.5,6.75), show=False)
            if self.save:
                plt.savefig(f'{path}{model_name}_violinplot.pdf')
            plt.clf()
            # plt.show()
            # clear_output()

    
    def get_sfs_importance(self, model_instance, model_name, X_train, y_train, path, N, fixed_N, metric):
        '''
        Function calculates RFE importances.
        Plot is made.
        '''
        # http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.feature_selection/#sequentialfeatureselector
        sfs = SFS(model_instance, 
                k_features=N, 
                forward=True, 
                floating=False, 
                fixed_features = None if fixed_N==None else list(range(fixed_N)),
                verbose=0,
                scoring=metrics[metric],
                n_jobs=-1,
                cv=StratifiedKFold(5))
        
        sfs = sfs.fit(X_train, y_train); clear_output()


        indexes = []
        for i in list(sfs.subsets_)[:0:-1]:
            indexes.append(list(set(sfs.subsets_[i]['feature_idx']) - set(sfs.subsets_[i-1]['feature_idx']))[0])
        
        if fixed_N == None:
            indexes.append(sfs.subsets_[i-1]['feature_idx'][0])

        feature_names = list(map((lambda x: eval(x)[1]), list(X_train.columns[indexes])))
        
        if fixed_N != None:
            feature_names.insert(0, 'clinical features')

        y_mean = [sfs.subsets_[i]['avg_score'] for i in list(sfs.subsets_)]
        metric_std = list(map((lambda x: np.std(x)), [sfs.subsets_[i]['cv_scores'] for i in list(sfs.subsets_)]))
        lower  = [(y_mean[i] - metric_std[i]) for i in list(range(len(y_mean)))]
        upper  = [(y_mean[i] + metric_std[i]) for i in list(range(len(y_mean)))]
        y_error = upper+lower[::-1]

        # save log
        data = pd.DataFrame(sfs.subsets_)
        data.loc["features"] = feature_names
        data.loc["std_score"] = metric_std
        if self.save:
            data.to_excel(f'{path}{model_name}_{metric}.xlsx')

        # plot graph
        if True:
            colors = ['0,100,80', '100,0,80']
            fig = go.Figure()

            # standard deviation area
            fig.add_traces(go.Scatter(x=feature_names+feature_names[::-1],
                                    y=y_error,
                                    fill='tozerox',
                                    line=dict(color=f'rgba({colors[0]},0.)'),
                                    showlegend=False,
            ))

            # line trace
            fig.add_traces(go.Scatter(x=feature_names,
                                    y=y_mean,
                                    line=dict(color=f'rgb({colors[0]})'),
                                    # showlegend=False,
                                    fillcolor=f'rgba({colors[0]},0.2)',
                                    name=f'{metric}',
                                    mode='markers+lines',
                                    ))


            if metric == 'roc_auc':
                fig.add_hline(y=0.5, line=dict(color='black',  width=2, dash='dot'), layer='below', row=i+1, col=1)
                fig.update_yaxes(range=[0., 1.], title='Metric score')
            else:
                fig.update_yaxes(range=[-1., 1.], title='Metric score')
            fig.update_layout(title=dict(text='Sequential forward feature selector', x=0.5))
            fig.update_layout(width=1300, height=450, margin=dict(l=40, r=40, t=40, b=40))
                    # update legend
            fig.update_layout(showlegend=True,
                                    legend=dict(orientation="h", 
                                                    title='Metric:',
                                                    yanchor="bottom",
                                                    y=1.001,
                                                    xanchor="left",
                                                    x=0.00, bordercolor="Grey",
                                                    borderwidth=0.5),)
            fig.show()

            if self.save:
                fig.write_image(f"{path}{model_name}_{metric}.pdf", engine="kaleido")
            return fig
        
    
    def get_rf_importance(self, model_instance, X_train, X_test, path, N=20, ):
        '''
        Function calculates Random Forest importances.
        Calculation is different for tree-based models and not-tree-based models.
        Bar plot is made.
        '''
        
        # save values
        feature_importances = pd.DataFrame(np.stack([X_train.columns, model_instance.feature_importances_], axis=1), columns=['features', 'values'])
        if self.save:
            feature_importances.to_excel(f'{path}values.xlsx')

        # plot graph
        if True:
            # sort and normalize
            feature_importances = feature_importances.sort_values("values", ascending=False, ignore_index=True)
            feature_importances['values'] = feature_importances['values']/feature_importances['values'][0]

            # get Bar plot
            fig = px.bar(
                        x='values',
                        orientation='h',
                        data_frame=feature_importances[:N][::-1],
                        y='features')
            
            # figure settings
            fig.update_layout(autosize=False, width=1000, height=450, margin=dict(l=60, r=20, t=60, b=40),) 
            fig.update_xaxes(title='Relative importance')
            fig.update_yaxes(title='')
            fig.update_layout(title=dict(text='Feature importance from Random Forest model', x=0.5,),)

            # fig.show(renderer='colab')
            if self.save:
                fig.write_image(f'{path}values.pdf', engine="kaleido")


    def get_importances(self, metric=None, fixed_N=None, N=None):
        '''
        Function for feature selection.
        Traget and dataset should be specified.
        Tuned parameters of model are read, and model is retrained.
        Chosen selectors are used to determine top N features. 
        '''

        for target in self.targets:

            for dataset in self.datasets:
                # links to preprocessed 'train' and 'test' data
                link_train = f'../Preprocessed Data/{target}/Imputed data/train_{dataset}.xlsx'
                link_test = f'../Preprocessed Data/{target}/Imputed data/test_{dataset}.xlsx'

                for subset in self.subsets:
                    # upload 'train' and 'test' data
                    X_train, X_test, y_train, y_test = self.get_train_test(link_train, link_test, dataset, subset, self.drop_col)

                    for model in self.models:
                        # get trained model instance
                        model_instance = self.get_model_instance(model=model, target=target, subset=subset, dataset=dataset, X_train=X_train, y_train=y_train) 

                        for selector in self.selectors:
                            # path to file with tuned parameters
                            results_path = f'../Feature selection/{target}/{subset} {dataset.upper()}/{self.folder}/{selector}/' # values.xslx, graph.pdf
                            if selector == 'SHAP':
                                self.get_shap_values_summary(model_instance=model_instance, model_name=model, X_train=X_train, X_test=X_test, path=results_path)
                            if selector == 'RFFI':
                                self.get_rf_importance(model_instance=model_instance, X_train=X_train, X_test=X_test, path=results_path)
                            if selector == 'SFS':
                                self.get_sfs_importance(model_instance=model_instance, model_name=model, X_train=X_train, y_train=y_train, path=results_path, N=N, fixed_N=fixed_N, metric=metric)
