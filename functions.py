from IPython.core.display import HTML

def normalize(x, col_max):
    if x == -1:
        return np.nan
    else:
        return x/col_max


def zscore(arr, window):
    x = arr.rolling(window = 1).mean()
    u = arr.rolling(window = window).mean()
    o = arr.rolling(window = window).std()

    return (x-u)/o


def _set_css_style(css_file_path):
   """
   Read the custom CSS file and load it into Jupyter.
   Pass the file path to the CSS file.
   """

   styles = open(css_file_path, "r").read()
   s = '<style>%s</style>' % styles
   return HTML(s)


# machine learning functions

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# define function to remove outliers

def remove_outliers(data, column):
    q_hi = data[column].quantile(0.999)
    q_low = data[column].quantile(0.001)

    data = data[(data[column] > q_low) & (data[column] < q_hi)]

    return data


# plotting functions

def make_bar_plot(data, column, title, xaxis, yaxis):
    temp = data[column].value_counts()
    temp_y0 = []
    temp_y1 = []

    for val in temp.index:
        temp_y1.append(np.sum(data["TARGET"][data[column] == val] == 1))
        temp_y0.append(np.sum(data["TARGET"][data[column] == val] == 0))

    zero = go.Bar(
        x=temp.index,
        y=(temp_y1 / temp.sum()) * 100,
        name='Zero',
        marker={'color': color_palette[3]}
    )

    one = go.Bar(
        x=temp.index,
        y=(temp_y0 / temp.sum()) * 100,
        name='One',
        marker={'color': color_palette[1]}
    )

    data = [one, zero]

    layout = go.Layout(
        title=title,
        width=1000,
        xaxis=dict(
            title=xaxis,
            tickfont=dict(
                size=14,
                color=color_palette[1]
            )
        ),
        yaxis=dict(
            title=yaxis,
            titlefont=dict(
                size=16,
                color=color_palette[2]
            ),
            tickfont=dict(
                size=14,
                color=color_palette[2]
            )
        ),
        legend=dict(
            orientation="h",
            yanchor="top"
        )
    )

    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(template="simple_white")

    return fig
