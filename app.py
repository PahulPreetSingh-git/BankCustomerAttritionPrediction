import pandas as pd
import dash
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import matplotlib as plt
import copy
from dash import no_update

from src.navbar import get_navbar
from src.graphs import df,layout,dist_plot,box_plot, scatter_plot, continuous_vars, cat_vars, df_new, svm_model, xgb_model, lr_model, rf_model
from src.graphs import df,layout,dist_plot,box_plot, scatter_plot,roc_plot,featureImportance_plot,accuracyResult,sensitivityResult,specificityResult,f1Result,aucResult

import plotly.express as px
from content import tab_prediction_features,tab_dataAnalysis_features,tab_modelAnalysis_features,tab_prediction_content
import time
import numpy as np


app = Dash(__name__,external_stylesheets = [dbc.themes.SUPERHERO,'/assets/styles.css'])
server = app.server

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

# Layout

tabs = dbc.Tabs(
    [
        dbc.Tab(tab_dataAnalysis_features, label="Data Analysis"),
        dbc.Tab(tab_modelAnalysis_features, label="Model Analysis"),
        dbc.Tab(tab_prediction_content, label="Prediction"),
        
    ]
)

jumbotron = dbc.Jumbotron(
    html.H2("Bank Customer Attrition Analysis and Prediction"),
    className="cover"
)

app.layout = html.Div([
    get_navbar(),
    jumbotron,
    # html.H4("Analysis and Prediction", className="cover"),
    html.Div(
        dbc.Row(dbc.Col(tabs, width=12)),
        id="mainContainer",
        style={"display": "flex", "flex-direction": "column"}
    ),
    html.P("Developed by Utkarsh Singh and Pahul Preet Singh", className="footer")
])

# callbacks

@app.callback(
    Output("categorical_pie_graph", "figure"),
    [
        Input("categorical_dropdown", "value"),
    ],
)

def donut_categorical(feature):

    time.sleep(0.2)

    temp = df.groupby([feature]).count()['Exited'].reset_index()

    fig = px.pie(temp, values="Exited", names=feature, hole=.5)

    layout_count = copy.deepcopy(layout)
    fig.update_layout(layout_count)
    
    _title = (feature[0].upper() + feature[1:]) + " Percentage"

    if(df[feature].nunique() == 2):
        _x = 0.3
    elif(df[feature].nunique() == 3):
        _x = 0.16
    else:
        _x = 0

    fig.update_layout(
        title = {'text': _title, 'x': 0.5},
        legend = {'x': _x}
    )

    return fig

@app.callback(
    Output("categorical_bar_graph", "figure"),
    [
        Input("categorical_dropdown", "value"),
    ],
)

def bar_categorical(feature):

    time.sleep(0.2)

    temp = df.groupby([feature, 'Exited']).count()['CustomerId'].reset_index()
    
    fig = px.bar(temp, x=feature, y="CustomerId",
             color=temp['Exited'].map({1: 'Churn', 0: 'NoChurn'}),
            #  color_discrete_map={"Churn": "#47acb1", "NoChurn": "#f26522"},
             barmode='group')
    
    layout_count = copy.deepcopy(layout)
    fig.update_layout(layout_count)
    
    _title = (feature[0].upper() + feature[1:]) + " Distribution by Churn"
    
    fig.update_layout(
        title = {'text': _title, 'x': 0.5},
        #xaxis_visible=False,
        xaxis_title="",
        yaxis_title="Count",
        legend_title_text="",
        legend = {'x': 0.16}
    )
    return fig


@app.callback(
    Output("numerical_density_graph", "figure"),
    [
        Input("numerical_dropdown", "value"),
    ],
)

def density_numerical(feature):
    time.sleep(0.2)
    fig=dist_plot(feature)
    return fig


@app.callback(
    Output("numerical_box_graph", "figure"),
    [
        Input("numerical_dropdown", "value"),
    ],
)

def box_numerical(feature):
    time.sleep(0.2)
    fig=box_plot(feature)
    return fig


@app.callback(
    Output("numerical_scatter_graph", "figure"),
    [
        Input("numerical_dropdown", "value"),
    ],
)

def scatter_numerical(feature):
    time.sleep(0.2)
    fig=scatter_plot(feature)
    return fig

#--------------------------------------------------
#Preidction Tab Starts
#--------------------------------------------------
@app.callback(
    [dash.dependencies.Output('lr_result', 'children'),
     dash.dependencies.Output('svm_result', 'children'),
     dash.dependencies.Output('rf_result', 'children'),
     dash.dependencies.Output('xgb_result', 'children')],
    [dash.dependencies.Input('btn_predict', 'n_clicks')],
    [dash.dependencies.State('ft_gender', 'value'),
     dash.dependencies.State('ft_geography', 'value'),
     dash.dependencies.State('ft_balance', 'value'),
     dash.dependencies.State('ft_estimatedsalary', 'value'),
     dash.dependencies.State('ft_age', 'value'),
     dash.dependencies.State('ft_hascreditcard', 'value'),
     dash.dependencies.State('ft_creditscore', 'value'),
     dash.dependencies.State('ft_numofproducts', 'value'),
     dash.dependencies.State('ft_isactivemember', 'value'),
     dash.dependencies.State('ft_tenure', 'value')]
)

def predict_churn(n_clicks, ft_gender, ft_geography, ft_balance, ft_estimatedsalary, ft_age,
                            ft_hascreditcard, ft_creditscore, ft_numofproducts, ft_isactivemember, ft_tenure):

    time.sleep(0.4)

    sample = {'Gender': ft_gender, 'Geography': ft_geography, 'Balance': float(ft_balance), 
    'EstimatedSalary': float(ft_estimatedsalary), 'Age': int(ft_age), 'HasCrCard': int(ft_hascreditcard),
    'CreditScore': int(ft_creditscore), 'NumOfProducts': int(ft_numofproducts), 
    'IsActiveMember': int(ft_isactivemember), 'Tenure': int(ft_tenure)
    }

    df_train = pd.DataFrame(sample, index=[0])
    df_train['TenureByAge'] = df_train.Tenure/(df_train.Age)
    df_train['BalanceSalaryRatio'] = df_train.Balance/df_train.EstimatedSalary
    df_train['CreditScoreGivenAge'] = df_train.CreditScore/(df_train.Age)
    
    df_train = df_train[continuous_vars + cat_vars]
    
    df_train.loc[df_train.HasCrCard == 0, 'HasCrCard'] = -1
    df_train.loc[df_train.IsActiveMember == 0, 'IsActiveMember'] = -1
    
    lst = ['Geography', 'Gender']
    remove = list()
    
    for i in lst:
        if (df_train[i].dtype == np.str or df_train[i].dtype == np.object):
            if i == 'Geography':
                for j in ['France', 'Spain', 'Germany']:
                    df_train[i+'_'+j] = np.where(df_train[i] == j,1,-1)
            
            elif i == 'Gender':
                for j in ['Male', 'Female']:
                    df_train[i+'_'+j] = np.where(df_train[i] == j,1,-1)
                
            remove.append(i)
    
    df_train = df_train.drop(remove, axis=1)
    df_train["id"] = 'New'

    df_train = pd.concat([df_new, df_train], ignore_index=True, sort=False)
    minVec = df_train[continuous_vars].min().copy()
    maxVec = df_train[continuous_vars].max().copy()
    df_train[continuous_vars] = (df_train[continuous_vars]-minVec)/(maxVec-minVec)
    
    df_train = df_train[df_train['id'] == 'New'].drop('id', axis=1)

    svm_prediction = svm_model.predict(df_train)
    xgb_prediction = xgb_model.predict(df_train)
    lr_prediction = lr_model.predict(df_train)
    rf_prediction = rf_model.predict(df_train)

    def churn_to_text(num):
        if(num == 0):
            return "Predicted: Not Churn"
        elif(num == 1):
            return "Predicted: Churn"

    # print(svm_prediction)

    if(n_clicks):
        return churn_to_text(lr_prediction), churn_to_text(svm_prediction), churn_to_text(rf_prediction), churn_to_text(xgb_prediction)
    else:
        return no_update

#--------------------------------------------------
#Preidction Tab Ends
#--------------------------------------------------


@app.callback(
    Output("model_ROC_graph", "figure"),
    [
        Input("model_dropdown", "value"),
    ],
)

def ROC_model(feature):
    time.sleep(0.2)
    fig=roc_plot(feature)
    return fig

@app.callback(
    Output("model_featureImportance", "figure"),
    [
        Input("model_dropdown", "value"),
    ],
)

def featuteImportance_model(feature):
    time.sleep(0.2)
    fig=featureImportance_plot(feature)
    return fig

@app.callback(
    Output("accuracy", "children"),
    [
        Input("model_dropdown", "value"),
    ],
)

def updateAccuracy(feature):
    time.sleep(0.2)
    return f"{round(accuracyResult(feature)*100,1)}%"

@app.callback(
    Output("sensitivity", "children"),
    [
        Input("model_dropdown", "value"),
    ],
)

def updateSensitivity(feature):
    time.sleep(0.2)
    return f"{round(sensitivityResult(feature)*100,1)}%"

@app.callback(
    Output("specificity", "children"),
    [
        Input("model_dropdown", "value"),
    ],
)

def updateSpecificity(feature):
    time.sleep(0.2)
    return f"{round(specificityResult(feature)*100,1)}%"

@app.callback(
    Output("f1", "children"),
    [
        Input("model_dropdown", "value"),
    ],
)

def updateF1(feature):
    time.sleep(0.2)
    return f"{round(f1Result(feature),3)}"


@app.callback(
    Output("auc", "children"),
    [
        Input("model_dropdown", "value"),
    ],
)

def updateAuc(feature):
    time.sleep(0.2)
    return f"{round(aucResult(feature),3)}"


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)