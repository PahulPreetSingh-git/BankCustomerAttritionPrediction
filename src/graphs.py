import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import copy
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
import pickle
from sklearn import metrics

# Model Read
svm_path = 'data/svm_model.sav'
with open(svm_path, "rb") as f:
    svm_Model = pickle.load(f)
    svm_accuracy = pickle.load(f)
    svm_f1=pickle.load(f)
    svm_specificity = pickle.load(f)
    svm_sensitivity = pickle.load(f)
    svm_auc = pickle.load(f)
    svm_feat_importances = pickle.load(f)
    svm_fpr = pickle.load(f)
    svm_tpr = pickle.load(f)

xgb_path = 'data/xgb_model.sav'
with open(xgb_path, "rb") as f:
    xgb_Model = pickle.load(f)
    xgb_accuracy = pickle.load(f)
    xgb_f1=pickle.load(f)
    xgb_specificity = pickle.load(f)
    xgb_sensitivity = pickle.load(f)
    xgb_auc = pickle.load(f)
    xgb_feat_importances = pickle.load(f)
    xgb_fpr = pickle.load(f)
    xgb_tpr = pickle.load(f)

lr_path = 'data/lr_model.sav'
with open(lr_path, "rb") as f:
    lr_Model = pickle.load(f)
    lr_accuracy = pickle.load(f)
    lr_f1=pickle.load(f)
    lr_specificity = pickle.load(f)
    lr_sensitivity = pickle.load(f)
    lr_auc = pickle.load(f)
    lr_feat_importances = pickle.load(f)
    lr_fpr = pickle.load(f)
    lr_tpr = pickle.load(f)

rf_path = 'data/rf_model.sav'
with open(rf_path, "rb") as f:
    rf_Model = pickle.load(f)
    rf_accuracy = pickle.load(f)
    rf_f1=pickle.load(f)
    rf_specificity = pickle.load(f)
    rf_sensitivity = pickle.load(f)
    rf_auc = pickle.load(f)
    rf_feat_importances = pickle.load(f)
    rf_fpr = pickle.load(f)
    rf_tpr = pickle.load(f)


layout = dict(
    autosize=True,
    #automargin=True,
    margin=dict(l=20, r=20, b=20, t=30),
    hovermode="closest",
    plot_bgcolor="#16103a",
    paper_bgcolor="#16103a",
    legend=dict(font=dict(size=10), orientation="h"),
    title="Satellite Overview",
    font_color ="#e0e1e6",
    xaxis_showgrid=False,
    yaxis_showgrid=False
)

# Model Read
svm_path = 'data/svm_model.sav'
with open(svm_path, "rb") as f:
    svm_model = pickle.load(f)

lr_path = 'data/lr_model.sav'
with open(lr_path, "rb") as f:
    lr_model = pickle.load(f)

rf_path = 'data/rf_model.sav'
with open(rf_path, "rb") as f:
    rf_model = pickle.load(f)

xgb_path = 'data/xgb_model.sav'
with open(xgb_path, "rb") as f:
    xgb_model = pickle.load(f)

# read data
df = pd.read_csv('data/bank_user_data.csv')

continuous_vars = ['CreditScore',  'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 
    'BalanceSalaryRatio', 'TenureByAge', 'CreditScoreGivenAge']
cat_vars = ['HasCrCard', 'IsActiveMember','Geography', 'Gender']

df_new = pd.read_csv('data/bank_user_data.csv')
df_new = df_new.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)
df_new['TenureByAge'] = df_new.Tenure/(df_new.Age)
df_new['BalanceSalaryRatio'] = df_new.Balance/df_new.EstimatedSalary
df_new['CreditScoreGivenAge'] = df_new.CreditScore/(df_new.Age)
df_new = df_new[continuous_vars + cat_vars]
lst = ['Geography', 'Gender']
remove = list()
for i in lst:
    if (df_new[i].dtype == np.str or df_new[i].dtype == np.object):
        for j in df_new[i].unique():
            df_new[i+'_'+j] = np.where(df_new[i] == j,1,-1)
        remove.append(i)
df_new = df_new.drop(remove, axis=1)
df_new["id"] = "Old"

cat_features = df.drop(['Geography','Gender', 'HasCrCard', 'IsActiveMember', 'Exited'],axis=1).columns

# Encoding categorical features
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse=False)

def dist_plot(feature):
    x1 = df[df['Exited'] == 0][feature].to_list()
    x2 = df[df['Exited'] == 1][feature].to_list()

    fig = ff.create_distplot([x1,x2],
    group_labels= ['NoChurn', 'Churn'],
    #  bin_size=3,
    curve_type='normal',
    show_rug=False,
                             show_hist=False,
                             show_curve=True,
                             colors=['#EF553B', '#636FFA']
                            )
    
    
    layout_count = copy.deepcopy(layout)
    fig.update_layout(layout_count)
    
    fig.update_layout(
        title = {'text': f"Density Plot of {feature}", 'x': 0.5},
        legend = {'x': 0.25}
    )
    

    return fig

def box_plot(feature):
    temp = df[['Exited', feature]]
    temp.loc[temp.Exited == 0, 'Exited'] = "NoChurn"
    temp.loc[temp.Exited == 1, 'Exited'] = "Churn"
    
    fig = px.box(temp, x='Exited', y=feature, 
    color='Exited',
    # color=df['Exited'].map({0: 'NoChurn', 1: 'Churn'}),
    # color_discrete_map={ "Churn": '#636FFA', "NoChurn": '#EF553B'}
    )
    
    layout_count = copy.deepcopy(layout)
    fig.update_layout(layout_count)
    
    fig.update_layout(
        title = {'text': f"Box plot of {feature}", 'x': 0.5},
        xaxis_title="",
        legend_title_text="",
        legend = {'x': 0.25}
    )
    

    return fig

def scatter_plot(feature):


    if feature == "Age":
        # temp = df[['Exited', feature, 'CreditScore']]
        # temp['Exited'] = temp['Exited'].astype(str)
        fig = px.scatter(df, x=feature, y='CreditScore',
        # color='Exited',
        # color_continuous_scale=[(0, "#636FFA"), (1, "#EF553B")],
        color=df['Exited'].map({0: 'NoChurn', 1: 'Churn'}),
        color_discrete_map={ "Churn": '#636FFA', "NoChurn": '#EF553B'}
        )
    else:
        # temp = df[['Exited', feature, 'Age']]
        # temp['Exited'] = temp['Exited'].astype(str)
        fig = px.scatter(df, x='Age', y=feature,
        # color='Exited',
        # color_continuous_scale=[(0, "#636FFA"), (1, "#EF553B")],
        color=df['Exited'].map({0: 'NoChurn', 1: 'Churn'}),
        color_discrete_map={ "Churn": '#636FFA', "NoChurn": '#EF553B'}
        )
    
    layout_count = copy.deepcopy(layout)
    fig.update_layout(layout_count)
    
    fig.update_layout(
        title = {'text': f"Scatter plot of {feature if feature != 'Age' else 'CreditScore'} vs Age", 'x': 0.5},
        xaxis_title="",
        legend_title_text="",
        legend = {'x': 0.25},
        # layout_coloraxis_showscale=False
    )
    

    return fig
    
def roc_plot(feature):
    if feature=='lr':
        print("LR")
        fpr=lr_fpr
        tpr=lr_tpr
    elif feature=='svm':
        print('SVM')
        fpr=svm_fpr
        tpr=svm_tpr
    elif feature=='rf':
        print("RF")
        fpr=rf_fpr
        tpr=rf_tpr
    else:
        print("XGB")
        fpr=xgb_fpr
        tpr=xgb_tpr

    score = metrics.auc(fpr, tpr)
    
    temp = pd.DataFrame(list(zip(fpr, tpr)),
               columns =['FPR', 'TPR'])
    temp['id'] = '1'
    fig = px.area(temp,
        x="FPR", y="TPR",
        color = "id",
        color_discrete_map={"1": '#EF553B'},
        labels=dict(
            x='False Positive Rate', 
            y='True Positive Rate'))
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1)

    layout_count = copy.deepcopy(layout)
    fig.update_layout(layout_count)

    fig.update_layout(
        title = {'text': f"ROC Curve", 'x': 0.5},
        showlegend=False,
    )

    return fig

def featureImportance_plot(feature):
    if feature=='lr':
        print("LR")
        featureImportance=lr_feat_importances
    elif feature=='svm':
        print('SVM')
        featureImportance=svm_feat_importances
    elif feature=='rf':
        print("RF")
        featureImportance=rf_feat_importances
    else:
        print("XGB")
        featureImportance=xgb_feat_importances

    featureImportance_Frame=featureImportance.to_frame().reset_index()

    fig = px.bar(featureImportance_Frame[::-1], x=0, y='index')
    layout_count = copy.deepcopy(layout)
    fig.update_layout(layout_count)
    fig.update_traces(marker_color='#EF553B')

    fig.update_layout(
        title = {'text': f"Features Importance", 'x': 0.5},
        xaxis_title="",
        yaxis_title="",
    )


    return fig

def accuracyResult(feature):
    if feature=='lr':
        return lr_accuracy
    elif feature=='svm':
        return svm_accuracy
    elif feature=='rf':
        return rf_accuracy
    else:
        return xgb_accuracy

def sensitivityResult(feature):
    if feature=='lr':
        return lr_sensitivity
    elif feature=='svm':
        return svm_sensitivity
    elif feature=='rf':
        return rf_sensitivity
    else:
        return xgb_sensitivity


def specificityResult(feature):
    if feature=='lr':
        return lr_specificity
    elif feature=='svm':
        return svm_specificity
    elif feature=='rf':
        return rf_specificity
    else:
        return xgb_specificity

def aucResult(feature):
    if feature=='lr':
        return lr_auc
    elif feature=='svm':
        return svm_auc
    elif feature=='rf':
        return rf_auc
    else:
        return xgb_auc

def f1Result(feature):
    if feature=='lr':
        return lr_f1
    elif feature=='svm':
        return svm_f1
    elif feature=='rf':
        return rf_f1
    else:
        return xgb_f1


        
        
