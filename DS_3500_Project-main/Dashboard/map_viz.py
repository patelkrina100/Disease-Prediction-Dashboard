"""
map_viz.py
Used to make map visualizations depicting the concentration and prevalance
of diseases across America
"""

import plotly.graph_objects as go


def make_map(df, measure, min_count=5000):
  """
  Creates a Bubble Map that shows the prevalence of the given measure across the USA

  :param df: given dataframe to pull data from
  :param measure: given measure representing a disease/statistic that will be displayed on the map
  :param min_count: the minimum number of cases for a given city in order to be shown on the map
  :return: a plotly figure that can then be passed to the show function
  """

  fig = go.Figure()
  df2 = df[df['Short_Question_Text'] == measure]
  df2 = df2[df2['TotalNumber'] >= min_count]
  df2['TotalNumText'] = df2['TotalNumber'].map(str)
  df2['text'] = df2['LocationName'] + '<br>Count ' + df2['TotalNumText']
  scale = 5000

  fig.add_trace(go.Scattergeo(
    locationmode = 'USA-states',
    lon = df2['Longitude'],
    lat = df2['Latitude'],
    text = df2['text'],
    marker = dict(
      size = df2['TotalNumber'] / scale,
      color = 'royalblue',
      line_color = 'rgb(40, 40, 40)',
      line_width = 0.5,
      sizemode = 'area'
    ),
    name = measure
  ))

  fig.update_layout(
    title_text = 'Prevalence by Location of {} in 2020 in the United States'.format(measure),
    showlegend = False,
    geo = dict(
      scope = 'usa',
      landcolor = 'rgb(217, 217, 217)',
    )
  )

  return fig


def make_map2(df, measure):
  """
    Creates a Heat Map that shows the prevalence of the given measure across the USA by state

    :param df: given dataframe to pull data from
    :param measure: given measure representing a disease/statistic that will be displayed on the map
    :return: a plotly figure that can then be passed to the show function
    """
  df2 = df[df['Short_Question_Text'] == measure]
  df2['Percent'] = df2.apply(lambda row: round((row.TotalNumber / row.ESTIMATESBASE2020), 4) * 100, axis=1)
  df2['Text'] = df2.apply(lambda row: row.StateDesc + '<br>' + \
    '{} found in '.format(measure) + '{}% of population'.format(row.Percent), axis=1)
  colorscale = 'reds'

  fig = go.Figure(data=go.Choropleth(
    locations=df2['StateAbbr'],
    z=df2['Percent'],
    locationmode = 'USA-states',
    colorscale=colorscale,
    colorbar={'title': 'Percentage of state population'},
    text=df2['Text'],
    marker={'line': {'color': 'white'}}
  ))

  fig.update_layout(
    title_text='Prevalence of {} by State in 2020'.format(measure),
    geo=dict(
      scope='usa'
    )
  )

  return fig