import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
import dash_html_components as html
from matplotlib.pyplot import margins

card_donut = dbc.Card(
    [
        dbc.CardBody(
            [
                dbc.Spinner(size="md",color="light",
                    children=[
                        dcc.Graph(id="categorical_pie_graph", config = {"displayModeBar": False}, style = {"height": "48vh"})
                    ]
                ),
                
            ],
        ),
    ],
    style = {"background-color": "#16103a"}
)

card_barChart= dbc.Card(
    [
        dbc.CardBody(
            [
                dbc.Spinner(size="md",color="light",
                    children=[
                        dcc.Graph(id="categorical_bar_graph", config = {"displayModeBar": False}, style = {"height": "48vh"})
                    ]
                ),
                
            ],
        ),
    ],
    style = {"background-color": "#16103a"}
)

card_boxPlot = dbc.Card(
    [
        dbc.CardBody(
            [
                dbc.Spinner(size="md",color="light",
                    children=[
                        dcc.Graph(id="numerical_box_graph", config = {"displayModeBar": False}, style = {"height": "48vh"})
                    ]
                ),
            ],
        ),
    ],
    style = {"background-color": "#16103a"}
)

card_densityPlot = dbc.Card(
    [
        dbc.CardBody(
            [
                dbc.Spinner(size="md",color="light",
                    children=[
                        dcc.Graph(id="numerical_density_graph", config = {"displayModeBar": False}, style = {"height": "48vh"})
                    ]
                ),
                
            ],
        ),
    ],
    style = {"background-color": "#16103a"}
)

card_scatter = dbc.Card(
    [
        dbc.CardBody(
            [
                dbc.Spinner(size="md",color="light",
                    children=[
                        dcc.Graph(id="numerical_scatter_graph", config = {"displayModeBar": False}, style = {"height": "48vh"})
                    ]
                ),
                
            ],
        ),
    ],
    style = {"background-color": "#16103a"}
)

card_ROC_Plot=dbc.Card(
    [
        dbc.CardBody([
                dbc.Spinner(size="md",color="light",
                    children=[
                        dcc.Graph(id="model_ROC_graph", config = {"displayModeBar": False}, style = {"height": "48vh"})
                    ]
                ),
        ])
    ],style = {"background-color": "#16103a"}
)

card_featureImportance=dbc.Card(
    [
        dbc.CardBody([
                dbc.Spinner(size="md",color="light",
                    children=[
                        dcc.Graph(id="model_featureImportance", config = {"displayModeBar": False}, style = {"height": "48vh"})
                    ]
                ),
        ])
    ],style = {"background-color": "#16103a"}
)

tab_modelAnalysis_features=html.Div([
    html.H4("Model Analysis", className="card-title"),
    html.Div([
        dbc.InputGroup([
            dbc.Select(                            
                options=[
                    {"label": "Logistic Regression", "value": "lr"},
                    {"label": "Support Vector Machine", "value": "svm"},
                    {"label": "Random Forest", "value": "rf"},
                    {"label": "XG Boost", "value": "xgb"},
                ], id = "model_dropdown", value="lr",
            )
        ], className="feature-row",),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Spinner(html.H1(id="accuracy", children="-", style={'color':'#e7b328'}), size="sm", spinner_style={'margin-bottom': '5px'}),
                        html.P("Accuracy")
                    ]
                ), className="result-card", style={"height":"16vh"}
            )
        ],lg=4, sm=6, className="card-padding"),
        dbc.Col([
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Spinner(html.H1(id="sensitivity", children="-", style={'color':'#e7b328'}), size="sm", spinner_style={'margin-bottom': '5px'}),
                        html.P("Sensitivity")
                    ]
                ), className="result-card", style={"height":"16vh"}
            )
        ],lg=4, sm=6, className="card-padding"),
        dbc.Col([
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Spinner(html.H1(id="specificity", children="-", style={'color':'#e7b328'}), size="sm", spinner_style={'margin-bottom': '5px'}),
                        html.P("Specificity")
                    ]
                ), className="result-card", style={"height":"16vh"}
            )
        ],lg=4, sm=6, className="card-padding"),
    ], className="feature-row",),
   
   dbc.Row([
        
        dbc.Col([
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Spinner(html.H1(id="auc", children="-", style={'color':'#e7b328'}), size="sm", spinner_style={'margin-bottom': '5px'}),
                        html.P("AUC")
                    ]
                ), className="result-card", style={"height":"16vh"}
            )
        ],lg=4, sm=6, className="card-padding"),
        
        dbc.Col([
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Spinner(html.H1(id="f1", children="-", style={'color':'#e7b328'}), size="sm", spinner_style={'margin-bottom': '5px'}),
                        html.P("F-1 Score")
                    ]
                ), className="result-card", style={"height":"16vh"}
            )
        ],lg=4, sm=6, className="card-padding"),
        
        dbc.Col([
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Spinner(html.H1(id="testTrainSplit", children="20% - 80%", style={'color':'#e7b328'}), size="sm", spinner_style={'margin-bottom': '5px'}),
                        html.P("Test - Train Split")
                    ]
                ), className="result-card", style={"height":"16vh"}
            )
        ],lg=4, sm=6, className="card-padding"),
    ], className="feature-row",),
    
    dbc.Row([
        dbc.Col(card_ROC_Plot, lg=6, sm=12),
        
        dbc.Col(card_featureImportance, lg=6, sm=12),
    ])
],className="mt-3", style = {"background-color": "#272953","padding":'10px'})

tab_dataAnalysis_features = html.Div(
            [
                html.H4("Categorical Attribute", className="card-title"),
                html.Div(
                    [
                    dbc.InputGroup([
                        dbc.Select(
                            options=[
                                {"label": "Gender", "value": "Gender"},
                                {"label": "Geography", "value": "Geography"},
                                {"label": "Has Credit Card", "value": "HasCrCard"},
                                {"label": "Is Active", "value": "IsActiveMember"},
                                {"label": "Number of Products", "value": "NumOfProducts"},
                            ], id = "categorical_dropdown", value="Gender",
                        ),
                    ],),
                    dbc.Card(
                            dbc.CardBody(
                                [
                                    dbc.Row([
                                        dbc.Col([
                                                html.Div([
                                                    html.Img(src="../assets/HeadExit.jpg", className="customer-img")
                                                ],),
                                        ], lg=4, sm=12,),
                                        dbc.Col(card_donut, lg="4", sm=12),
                                        dbc.Col(card_barChart, lg="4", sm=12),
                    ]),

                                ],
                            ), className="mt-3", style = {"background-color": "#272953"},
                    ),
                    # dbc.Row([
                    #     dbc.Col([
                    #             html.Div([
                    #                 html.Img(src="../assets/HeadExit.jpg", className="customer-img")
                    #             ],),
                    #     ],lg="4", sm=12,style = {"background-color": "#16103a"}),
                    #     dbc.Col(card_donut, lg="4", sm=12),
                    #     dbc.Col(card_barChart, lg="4", sm=12),
                    # ]),
                ],className="feature-row",
                ),
                html.H4("Numerical Attribute", className="card-title"),
                    html.Div(
                        [
                        dbc.InputGroup([
                            dbc.Select(
                                options=[
                                    {"label": "Credit Score", "value": "CreditScore"},
                                    {"label": "Balance", "value": "Balance"},
                                    {"label": "Estimated Salary", "value": "EstimatedSalary"},
                                    {"label": "Age", "value": "Age"},
                                ], id = "numerical_dropdown", value="CreditScore",
                            ),
                        ],),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dbc.Row([
                            dbc.Col(card_densityPlot, lg="4", sm=12),
                            dbc.Col(card_scatter, lg="4", sm=12),
                            dbc.Col(card_boxPlot, lg="4", sm=12),
                        ]),

                                ],
                            ), className="mt-3", style = {"background-color": "#272953"},
                        ),
                        # dbc.Row([
                        #     dbc.Col(card_densityPlot, lg="4", sm=12),
                        #     dbc.Col(card_scatter, lg="4", sm=12),
                        #     dbc.Col(card_boxPlot, lg="4", sm=12),
                        # ]),
                    ],
                    ),
            ]
        ),

tab_prediction_features = dbc.Card(
    dbc.CardBody(
        [
            # First Row

            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupAddon("Gender", addon_type="prepend"),
                                    dbc.Select(
                                        id="ft_gender",
                                        options=[
                                            {"label": "Female", "value": "Female"},
                                            {"label": "Male", "value": "Male"},
                                        ], value="Female",className="custom-select-new"
                                    )
                                ]
                            )
                        ], lg="6", sm=12
                    ),

                    dbc.Col(
                        [
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupAddon("Geography", addon_type="prepend"),
                                    dbc.Select(
                                        id="ft_geography",
                                        options=[
                                            {"label": "France", "value": "France"},
                                            {"label": "Germany", "value": "Germany"},
                                            {"label": "Spain", "value": "Spain"}
                                        ], value="France", className="custom-select-new"
                                    )
                                ]
                            )
                        ], lg="6", sm=12
                    ),  
                ], className="feature-row",
            ), 

            # Second Row

            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupAddon("Balance", addon_type="prepend"),
                                    dbc.Input(
                                        id="ft_balance",
                                        placeholder="Balance", type="number", value="2100"
                                    ),
                                ]
                            )
                        ], lg="6", sm=12
                    ),

                    dbc.Col(
                        [
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupAddon("Estimated Salary", addon_type="prepend"),
                                    dbc.Input(
                                        id="ft_estimatedsalary",
                                        placeholder="Estimated Salary", type="number", value="110000"
                                    ),
                                ]
                            )
                        ], lg="6", sm=12
                    ),
                    
                ], className="feature-row",
            ),

            # Third Row

            dbc.Row(
                [
                    # Multiple Lines

                    dbc.Col(
                        [
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupAddon("Age", addon_type="prepend"),
                                    dbc.Input(
                                        id="ft_age",
                                        placeholder="Age", type="number", value="25"
                                    ),
                                ]
                            )
                        ], lg="6", sm=12
                    ),

                    dbc.Col(
                        [
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupAddon("Has Credit Card", addon_type="prepend"),
                                    dbc.Select(
                                        id="ft_hascreditcard",
                                        options=[
                                            {"label": "Yes", "value": "1"},
                                            {"label": "No", "value": "0"},
                                        ], value="0", className="custom-select-new"
                                    )
                                ]
                            )
                        ], lg="6", sm=12
                    ),
                ], className="feature-row",
            ),

            # Fourth Row

            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupAddon("Credit Score", addon_type="prepend"),
                                    dbc.Input(
                                        id="ft_creditscore",
                                        placeholder="Credit Score", type="number", value="300"
                                    ),
                                ]
                            )
                        ], lg="6", sm=12
                    ),

                    dbc.Col(
                        [
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupAddon("Number of Products", addon_type="prepend"),
                                    dbc.Input(
                                        id="ft_numofproducts",
                                        placeholder="Number of Products", type="number", value="1"
                                    ),
                                ]
                            )
                        ], lg="6", sm=12
                    ),
                ], className="feature-row",
            ),

            # Fifth Row

            dbc.Row(
                [
                    

                    dbc.Col(
                        [
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupAddon("Is Active Member", addon_type="prepend"),
                                    dbc.Select(
                                        id="ft_isactivemember",
                                        options=[
                                            {"label": "Yes", "value": "1"},
                                            {"label": "No", "value": "0"},
                                        ], value="0", className="custom-select-new"
                                    )
                                ]
                            )
                        ], lg="6", sm=12
                    ), 

                    dbc.Col(
                        [
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupAddon("Tenure", addon_type="prepend"),
                                    dbc.Input(
                                        id="ft_tenure",
                                        placeholder="Tenure", type="number", value="3"
                                    ),
                                ]
                            )
                        ], lg="6", sm=12
                    ), 
                ]
            )
        ]
    ),
    className="mt-3", style = {"background-color": "#272953"}
)

tab_prediction_result = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Button("Predict", id='btn_predict', size="lg", className="btn-predict")
                        ], lg=12, sm=12, style={"display": "flex", "align-items":"center", "justify-content":"center"},
                        className="card-padding"
                    )
                ], className="feature-row",
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Spinner(html.H3(id="lr_result", children="-", style={'color':'#e7b328'}), size="sm", spinner_style={'margin-bottom': '5px'}),
                                        html.P("Logistic Regression")
                                    ]
                                ), className="result-card", style={"height":"16vh"}
                            )
                        ], lg=3, sm=3, className="card-padding-new"
                    ),

                    dbc.Col(
                        [
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Spinner(html.H3(id="svm_result", children="-", style={'color':'#e7b328'}), size="sm", spinner_style={'margin-bottom': '5px'}),
                                        html.P("SVM")
                                    ]
                                ), className="result-card", style={"height":"16vh"}
                            )
                        ], lg=3, sm=3, className="card-padding"
                    ),

                    dbc.Col(
                        [
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Spinner(html.H3(id="rf_result", children="-", style={'color':'#e7b328'}), size="sm", spinner_style={'margin-bottom': '5px'}),
                                        html.P("Random Forest")
                                    ]
                                ), className="result-card", style={"height":"16vh"}
                            )
                        ], lg=3, sm=3, className="card-padding"
                    ),

                    dbc.Col(
                        [
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Spinner(html.H3(id="xgb_result", children="-", style={'color':'#e7b328'}), size="sm", spinner_style={'margin-bottom': '5px'}),
                                        html.P("XGBoost")
                                    ]
                                ), className="result-card", style={"height":"16vh"}
                            )
                        ], lg=3, sm=3, className="card-padding"
                    )


                ]
            ),
        ]
    ),
    className="mt-3", style = {"background-color": "#272953"}
)

tab_prediction_content = [
    
    tab_prediction_features,
    tab_prediction_result
    
]