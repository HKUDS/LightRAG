"""
Postprocessing functions for cleaning up latex equations in linear format which don't give valid LaTeX.
"""
import re

clean_exps = {
    r"\\degf": "&deg;F",
    r"\\degc": "&deg;C",
    r"(\\cbrt)(\w+)": r"\\sqrt[3]{\2}",
    r"(\\qdrt)(\w+)": r"\\sqrt[4]{\2}",
    r"\\sfrac": r"\\frac",
    r"(\\o[i]+nt)(\w+)": r"\1{\2}",
    r"\\bullet(\w+)": r"\\bullet \1",
    r"\\sum([a-zA-Z0-9]+)": r"\\sum{\1}",
    r"\\prod([a-zA-Z0-9]+)": r"\\prod{\1}",
    r"\\amalg([a-zA-Z0-9]+)": r"\\amalg{\1}",
    r"\\bigcup([a-zA-Z0-9]+)": r"\\bigcup{\1}",
    r"\\bigcap([a-zA-Z0-9]+)": r"\\bigcap{\1}",
    r"\\bigvee([a-zA-Z0-9]+)": r"\\bigvee{\1}",
    r"\\bigwedge([a-zA-Z0-9]+)": r"\\bigwedge{\1}",
    r"\\lfloor([a-zA-Z0-9]+)": r"\\lfloor{\1}",
    r"\\lceil([a-zA-Z0-9]+)": r"\\lceil{\1}",
    r"\\lim\\below\{(.+)\}\{(.+)\}": r"\\lim_{\1}{\2}",
    r"\\min\\below\{(.+)\}\{(.+)\}": r"\\min_{\1}{\2}",
    r"\\max\\below\{(.+)\}\{(.+)\}": r"\\max_{\1}{\2}",
}


def clean_exp(exp):
    """
    Takes in a linear expression and converts known invalid LaTeX equations to valid LaTeX
    :param exp:str - An equation in invalid syntax
    :return :str - A valid equation
    """
    for e in clean_exps:
        exp = re.sub(e, clean_exps[e], exp)
    return exp
