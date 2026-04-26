import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    __[Delete this text and add team name, as well as names of all team members]__

    # Project Title

    _Brief and informative, gives some idea of your topic area_
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Introduction

    _Include core background information and ethical considerations relevant to initial data collection and contemporary use of the data. Articulate a research question that can be addressed with computational, data-driven analysis. Describe your focal data set and research design._
    """)
    return


@app.cell
def _():
    # code and /or markdown here as needed
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Methods

    _Describe your methods and why they are appropriate for the research question. Describe the strengths and weaknesses of all methods to be used. Cite sources as needed to justify how the methods are to be used and interpreted._
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Results

    _Fully report your results, including the relevant predictive power, statistical significance, and/or validity of all implemented models. Discuss coefficients where relevant. Use tables and figures as needed._
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation

    _This section should include a fully developed interpretation that is consistent with the results and clearly addresses the research question. Discuss here any major caveats or limitations to the interpretation, the extent to which it can be generalized, and how it might be extended by further research._
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Uses of Python: Reflection

    _Take a step back and analyze your own use of code. Includes table of technical dependencies. Provide some rationale for choices you’ve made. Considerations may include performance, human readability, code dependencies, and reproducibility._
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## References

    _List all works cited in the data guide. Use proper APA format._
    """)
    return


if __name__ == "__main__":
    app.run()
