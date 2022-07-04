import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.datasets import fetch_california_housing
import plotly.graph_objects as go

def main():
    # Importing Data
    st.header("Dataset")
    cal_housing = fetch_california_housing()
    X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    y = cal_housing.target

    st.dataframe(X)
    df = pd.DataFrame(
        dict(MedInc=X['MedInc'], Price=cal_housing.target))

    st.dataframe(df)

    st.subheader("House Age independent General Model")
    fig = px.scatter(df, x="MedInc", y="Price")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Note:")
    st.write(fr"""If there are such *Beta* values that make all the errors in the loss function higher than the threshold,
     the function becomes constant and the gradient becomes zero. In order to prevent this, there will be a number of **pre-searches**
     to find the *Beta* values which has the minimum error and we will start with that *Beta* values.""")
    # The range of pre-search
    rng = st.slider("The Number of Iterations for The Pre-Search:", 1, 200, value=100)
    threshold = st.slider("Threshold for The Maximum Error:", 0.1, 15., value=3.)
    lam_l2 = st.slider("Regularization Multiplier for L2 (lambda):", 0.001, 5., value=0.1)
    st.subheader("")

    ### Gradient Descent ###
    beta_gd, y_pred_list_reg = ls_gd(df['MedInc'].values, y, threshold, rng)

    st.subheader("Gradient Descent")
    st.latex(fr"Price = {beta_gd[1]:.4f} \times MedInc + {beta_gd[0]:.4f}")
    st.subheader("")

    ### L2 Regularization ###
    beta_l2 = ls_l2(df['MedInc'].values, y, lam_l2, threshold, rng)

    st.subheader("L2 Regularization")
    st.latex(fr"Price = {beta_l2[1]:.4f} \times MedInc + {beta_l2[0]:.4f}")

    ### Comparison Graph ###
    fig.add_trace(
        go.Scatter(
            x = df['MedInc'].values,
            y = -0.0524 + 1.1100 * (df['MedInc'].values),
            mode = "lines",
            name = "We Do Model"
        )
    )

    fig.add_trace(
        go.Scatter(
            x = df['MedInc'].values,
            y = beta_gd[0] + beta_gd[1] * (df['MedInc'].values),
            mode = "lines",
            name = "Gradient Descent"
        )
    )

    fig.add_trace(
        go.Scatter(
            x = df['MedInc'].values,
            y = beta_l2[0] + beta_l2[1] * (df['MedInc'].values),
            mode = "lines",
            name = "L2 Regularized"
        )
    )

    st.subheader("")
    st.subheader("Comparison")
    st.plotly_chart(fig, use_container_width=True)
    st.write(fr"**Note:** In We Do section, we've found *Beta_0 = -0.0524* and *Beta_1 = 1.1100* with %80 contribution of general model")

### GRADIENT DESCENT FUNCTION###
def ls_gd(x, y, thr, rng, alpha=0.000001):
    y_pred = 0
    iter_no = 0
    min_error = 1000000000
    beta_min = []
    gd_bar = st.progress(0)
    # Pre-search starts
    for iter in range(rng):
        gd_bar.progress(iter/100)
        beta = np.random.random(2)
        error = 0
        y_pred_list = []

        for _x, _y in zip(x, y):
            y_pred = beta[0] + beta[1] * _x
            y_pred_list.append(y_pred)

            if (np.abs(_y - y_pred) >= thr):
                error += thr
            else:
                error += (_y - y_pred) ** 2

        if error < min_error:
            min_error = error
            beta_min = beta
            iter_no = iter

        print(f"({iter})  Beta = [ {beta} ]  Error = {error}")

    print(f"We will start with Beta = [ {beta_min} ] & Error = {min_error} at iteration {iter_no}")
    # Pre-search ends
    print("\n\nStarting Gradient Descent")

    for i in range(500):
        y_pred_gd: np.ndarray = beta_min[0] + beta_min[1] * x

        g_b0 = -2 * (y - y_pred_gd).sum()
        g_b1 = -2 * (x * (y - y_pred_gd)).sum()

        print(f"({i}) Beta: [ {beta} ], Gradient: [ {g_b0} , {g_b1} ]")

        beta_prev = np.copy(beta_min)

        beta_min[0] = beta_min[0] - alpha * g_b0
        beta_min[1] = beta_min[1] - alpha * g_b1

        if np.linalg.norm(beta_min - beta_prev) < 0.00005:
            print(f"I do early stoping at iteration {i}")
            break

    return beta_min, y_pred_gd

### L2 REGULARIZED FUNCTION ###
def ls_l2(x, y, lam, thr, rng, alpha=0.000001) -> np.ndarray:
    y_pred = 0
    min_error = 1000000000
    beta_min = []
    l2_iter = st.progress(0)
    # Pre-search starts
    for iter in range(rng):
        l2_iter.progress(iter/100)
        beta = np.random.random(2)
        error = 0
        y_pred_list = []

        for _x, _y in zip(x, y):
            y_pred = beta[0] + beta[1] * _x
            y_pred_list.append(y_pred)

            if (np.abs(_y - y_pred) >= thr):
                error += thr
            else:
                error += (_y - y_pred) ** 2

        if error < min_error:
            min_error = error
            beta_min = beta
    # Pre-search ends
    print("\n\nStarting L2 Regularized")
    # Initialize beta with the minimum error found in pre-search
    beta = beta_min

    for i in range(1000):
        y_pred: np.ndarray = beta[0] + beta[1] * x

        g_b0 = -2 * (y - y_pred).sum() + 2 * lam * beta[0]
        g_b1 = -2 * (x * (y - y_pred)).sum() + 2 * lam * beta[1]

        print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}")

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        if np.linalg.norm(beta - beta_prev) < 0.000001:
            print(f"I do early stoping at iteration {i}")
            break

    return beta


if __name__ == '__main__':
    main()
