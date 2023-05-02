"""
A dashboard that predicts the disease of a user based on their symptoms and reports to the user some useful
information regarding various disease statistics across the United States
"""
from disease_predictor import *
from sankey_disease_syms import *
from map_viz import *
import pandas as pd
from dash import Dash, dcc, html, Input, Output
from dash.exceptions import PreventUpdate
pd.options.mode.chained_assignment = None

# Read in the data for the disease prediction
initial_df = pd.read_csv('disease_data.csv')

# Obtain the disease and symptoms DataFrame and a list of the symptoms in the data
dis_sym_df, sym_lst = get_prediction_df(initial_df)

# Create a DataFrame to use to produce our sankey diagram
sankey_df = make_df(initial_df)

# Obtain a list of all the diseases in our prediction data
disease = list(sankey_df['Disease'].unique())

# Create a list of options for the user to make a sankey diagram
sk_options = ['All'] + disease

# Read in the precaution data
prec_df = pd.read_csv('symptom_precaution.csv')

# Get the data needed to create the heat map
heat_df = pd.read_csv('grouped_2020.csv')

# Get the data needed to create the disease prevalence map
dis_prev_df = pd.read_csv('measures_2020.csv')

# Obtain a list of the diseases that can be used to create our disease prevalance map
disease_unique = list(dis_prev_df.Short_Question_Text.unique())

# Build an app to display sunspot data
app = Dash(__name__)

# Due to the implementation of tabs, suppress callback exceptions
app.config.suppress_callback_exceptions = True

# Define the components needed to style the dashboard
style = {'background': '#270980', 'text': '#270980', 'font-family': 'candara'}
# OTHER POTENTIAL COLORS: 7AD6EB or white

# Format the layout of the dashboard
app.layout = html.Div(style={'textAlign': 'center', 'fontWeight': 'bold'},
                      children=[
    html.H1('Interactive Dashboard for Disease Prediction and Disease Information Reporting', style={'backgroundColor':
                                                                                                         '#270980',
                                                                                                     'margin': '0',
                                                                                                     'color': 'white',
                                                                                                     'font-size': '200%'
                                                                                                     'padding-top:10px'}),
    dcc.Tabs(id='tabs', value='tab_1', children=[
        dcc.Tab(label='Introduction', value='tab_1', style={'color': style['text'], 'font-family': style['font-family'],
                                                            'background': 'white'}),
        dcc.Tab(label='Disease Prediction', value='tab_2', style={'color': style['background'], 'font-family': style['font-family'],
                                                                  'background':'white'}),
        dcc.Tab(label='Sankey Diagrams', value='tab_3', style={'color': style['background'], 'font-family': style['font-family'],
                                                               'background':'white'}),
        dcc.Tab(label='Disease Heat Map', value='tab_4', style={'color': style['background'], 'font-family': style['font-family'],
                                                          'background':'white'}),
        dcc.Tab(label='Disease Prevalence Map', value='tab_5', style={'color': style['background'], 'font-family': style['font-family'],
                                                                      'background':'white'})
    ]),
    html.Div(id='tabs_content')
])


# Make a decorator for the tabs and render the content of the tabs
@app.callback(
    Output('tabs_content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab_1':
        return html.Div([
            html.Img(src=app.get_asset_url('intro_img.png'), style={'margin': 10}, width=400, height=300),
            html.P('Welcome to the interactive dashboard for disease prediction and disease information reporting. \
                       Our goal is to inform you in regards to your personal medical concerns. This dashboard contains \
                       several interactive tabs which will allow you to gain a better understanding of your diagnosis, as \
                       well as understand general disease statistics within the United States.', style={'margin-left': 100,
                                                                                                        'margin-right':100}),
            html.P('This Dashboard performs several tasks. A Disease Prediction model allows you to insert your symptoms \
                       and obtain a diagnosis along with a prescribed treatment. Both a heat map and bubble map display \
                    disease prevalence based on a chosen disease. Finally, a Sankey diagram links symptoms to diseases and vice-versa \
                       to show the frequency at which symptoms and diseases are correlated, and which diseases share the same \
                       symptoms.', style={'margin-left': 100, 'margin-right':100}),
            html.H3('Data Sources', style={'text-align': 'left', 'font-family': style['font-family'],
                                           'padding-left': '65px'}),
            html.Li(html.A('Disease Prediction and Sankey Diagram Data',
                           href="https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset",
                           target='_blank'),
                    style={'text-align': 'left', 'font-family': style['font-family'], 'font-size':'110%',
                           'padding-left': '75px'}),
            html.Li(html.A('Map Data',
                           href="https://chronicdata.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-\
                           Place-Data-202/eav7-hnsx",
                           target='_blank'),
                    style={'text-align': 'left', 'font-family': style['font-family'], 'font-size':'110%',
                           'padding-left': '75px'})
        ])
    elif tab == 'tab_2':
        return html.Div(children=[
            html.P('Please select the symptoms you are experiencing:'),
            dcc.Dropdown(sym_lst, multi=True, id='options_chosen'),
            html.P('Please select the algorithm you would like to use for this prediction:'),
            dcc.RadioItems(options=['Random Forest Classifier', 'Naive Bayes Model', 'K-Nearest Neighbors Classifier',
                                    'Logistic Regression Model'],
                           value='Random Forest Classifier',
                           id='radio_button',
                           style={'font-family': style['font-family'], 'font-size':'120%'}),
            html.P('Please press the done button when you have finished entering your symptoms'),
            dcc.Checklist(['Done'], id='checklist', style={'font-family':style['font-family']}),
            dcc.Loading(id='loading', type='default', children=html.Div(id='text_return')),
            html.Div(id='text_return'),
            html.P('To receive a new prediction, please refresh the page')
        ])
    elif tab == 'tab_3':
        return html.Div([
            html.P('Please select the disease(s) you would like to see linked to their corresponding symptoms:'),
            dcc.Dropdown(sk_options, multi=True, id='disease_chosen'),
            dcc.Graph(id='sankey', style={'width': '100vw', 'height': '65vh'})

        ])
    elif tab == 'tab_4':
        return html.Div([
          html.P('Please select the disease you would like to see a heat map of:'),
          dcc.Dropdown(disease_unique, id='disease_list'),
          dcc.Graph(id='dis_heat_map', style={'width': '100vw', 'height': '77vh'})
        ])
    elif tab == 'tab_5':
        return html.Div([
            html.P('Please select the disease you would like to see a prevalence map of:'),
            dcc.Dropdown(disease_unique, id='disease_unique'),
            dcc.Graph(id='dis_prev_map', style={'width': '100vw', 'height': '65vh'})
        ])

# Create a decorator for tab 2: disease predictor
@app.callback(
    Output('text_return', 'children'),
    Input('options_chosen', 'value'),
    Input('checklist', 'value'),
    Input('radio_button', 'value'),
    Input('checklist', 'value')
)
def update_predictor(options_chosen, checklist, radio_button, value):
    # Prevent updates to the dash if the user is not done uploading their symptoms
    if not options_chosen:
        raise PreventUpdate
    if not checklist:
        raise PreventUpdate

    # The most important features of the disease_data as determined by the code in feature_selection.py (used to prevent over-fitting)
    imp_features = ['abdominal_pain', 'altered_sensorium', 'back_pain', 'belly_pain', 'breathlessness', 'chest_pain',
                    'chills', 'dark_urine', 'dehydration', 'diarrhoea', 'dischromic _patches', 'family_history',
                    'fast_heart_rate', 'fatigue', 'headache', 'high_fever', 'internal_itching', 'joint_pain',
                    'lack_of_concentration', 'malaise', 'mild_fever', 'mucoid_sputum', 'muscle_pain', 'muscle_weakness',
                    'nausea', 'pain_behind_the_eyes', 'stomach_bleeding', 'stomach_pain', 'sweating', 'unsteadiness',
                    'vomiting', 'weight_loss', 'yellowing_of_eyes', 'itching']

    # Update the disease and symptoms dataframe to consider only important features of the data
    pred_df = dis_sym_df[imp_features]

    # Create a list of 0s and 1s that corresponds to whether the user is feeling certain symptoms (1) or not (0)
    user_syms = []
    appender = [user_syms.append(0.0) if sym not in options_chosen else user_syms.append(1.0) for sym in
                list(pred_df.columns)[0:]]

    # Create a DataFrame using the user given symptoms
    user_syms_df = pd.DataFrame(data=[user_syms], columns=list(pred_df.columns)[0:])

    # Train each model and get the accuracy of each
    rfc_modl, nb_modl, knn_modl, lr_modl, rfc_cv_score, nb_cv_score, knn_cv_score, lr_cv_score = \
        predict_disease(initial_df, imp_features)

    # Create the prediction for the user based on their algorithm choice
    if radio_button == 'Random Forest Classifier':
        pred_disease = str(rfc_modl.predict(user_syms_df)[0])
        cross_val_acc = rfc_cv_score
    elif radio_button == 'K-Nearest Neighbors Classifier':
        pred_disease = str(knn_modl.predict(user_syms_df)[0])
        cross_val_acc = knn_cv_score
    elif radio_button == 'Logistic Regression Model':
        pred_disease = str(lr_modl.predict(user_syms_df)[0])
        cross_val_acc = lr_cv_score
    else:
        pred_disease = str(nb_modl.predict(user_syms_df)[0])
        cross_val_acc = nb_cv_score

    # Obtain the precautions the user should take given their symptoms
    prec_str = report_precautions(pred_disease, prec_df)

    # Create a markdown to return all this information to the user
    markdown = dcc.Markdown(
        f"""
    **Your Predicted Disease:** {str(pred_disease)}
    
    **Accuracy of your chosen prediction algorithm:** {str(cross_val_acc)}%
    
    **Your recommended precautions:** {prec_str}
    
    ## Notice: the diagnosis you have obtained from this site is merely a *prediction*. Please consult a doctor before experiencing concern and/or seeking any medical treatment. 
    
        """, id='text_return', style={'margin':30, 'font-family': style['font-family'], 'color': style['text']})

    return markdown


# Create a decorator for tab 3: sankey diagram
@app.callback(
    Output('sankey', 'figure'),
    Input('disease_chosen', 'value'),
)
def update_sankey(disease_chosen):
    # Prevent updates if the user has not chosen a disease
    if not disease_chosen:
        raise PreventUpdate

    # Account for the instance where the user wants to see all diseases paired to corresponding symptoms
    if disease_chosen == ['All']:
        disease_chosen = disease

    # Filter the data based off the disease(s) the user chooses
    df_input = filter_input(sankey_df, 'Disease', disease_chosen)

    # Make a sankey diagram to show the user chosen disease(s) paired to symptoms
    fig = make_sankey_diagrams(df_input, 'Disease', 'Symptom', 'Val')

    return fig

# Create a decorator for tab 4: disease heat map
@app.callback(
    Output('dis_heat_map', 'figure'),
    Input('disease_list', 'value'),
)
def update_state_prev_fig(disease_unique):
    # Fig returned corresponds to filtered disease
    heat_fig = make_map2(heat_df, disease_unique)
    return heat_fig

# Create a decorator for tab 5: disease prevalence diagram
@app.callback(
    Output('dis_prev_map', 'figure'),
    Input('disease_unique', 'value'),
)
def update_city_prev_fig(disease_unique):
    # Fig returned corresponds to filtered disease
    prevalence_fig = make_map(dis_prev_df, disease_unique)
    return prevalence_fig


if __name__ == '__main__':
    # Run the app
    app.run_server(debug=True)