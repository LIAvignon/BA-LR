# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================

from bokeh.palettes import inferno,Category20
from bokeh.plotting import figure, show
from bokeh.io import output_file, show,output_notebook
from bokeh.transform import factor_cmap



def plot_family_bars(df_plot):
    df_plot.Members = df_plot.Members.astype(str)
    df_plot.Family = df_plot.Family.astype(str)
    group = df_plot.groupby(['Family', 'Members'])
    #source = ColumnDataSource(data=group)
    index_cmap = factor_cmap('Family_Members', palette=Category20[18], factors=df_plot.Family.unique(), end=1)
    p = figure(width=1200, height=600, title="Contribution of variables grouped by families",
               x_range=group, toolbar_location=None,tooltips=[("Cont_Member", "@Cont_Member_mean"),("Cont_Family", "@Cont_Family_mean")])
    p.vbar(x='Family_Members', top='Cont_Member_mean', width=1, source=group,line_color="white", fill_color=index_cmap, )
    p.y_range.start = 0
    p.xgrid.grid_line_color = None
    p.xaxis.axis_label = ""
    p.outline_line_color = None
    p.xaxis.major_label_orientation = "vertical"
    output_file(f"data/BA/bar_cont_{ba}.html")
    show(p)
def plot_bar_BA(df_cont_bas):
    df_cont_bas.BA = df_cont_bas.BA.astype(str)
    df_cont_bas.Family = df_cont_bas.Family.astype(str)
    group = df_cont_bas.groupby(['Family', 'BA'])
    index_cmap = factor_cmap('Family_BA', palette=inferno(len(df_cont_bas.BA.unique())), factors=df_cont_bas.BA.unique(), start=1,end=2)
    p = figure(width=2000, height=300, title="Contribution of BAs to each family",
               x_range=group, toolbar_location=None)
    p.vbar(x='Family_BA', top='Con_Fam_mean', width=1, source=group,line_color="white", fill_color=index_cmap, )
    p.y_range.start = 0
    p.xgrid.grid_line_color = None
    p.xaxis.axis_label = ""
    p.outline_line_color = None
    p.xaxis.major_label_orientation = "vertical"
    output_file(f"data/BA/BAs_cont.html")
    show(p)