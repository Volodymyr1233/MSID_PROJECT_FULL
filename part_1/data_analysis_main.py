from create_boxplot import execute_create_boxplot
from create_violinplots import execute_create_violinplots
from create_histogram import execute_histogram
from create_histogram_cond import execute_create_diagram_cond
from create_errorbar import execute_errorBar
from create_heatmap import create_heatmap_age_financil
from create_linear_regr import execute_regression
from get_info_kateg import get_info_kateg
from get_info import get_info


if __name__ == "__main__":
    get_info_kateg()
    get_info()
    execute_create_boxplot()
    execute_create_violinplots()
    execute_histogram()
    execute_create_diagram_cond()
    execute_errorBar()
    create_heatmap_age_financil()
    execute_regression()