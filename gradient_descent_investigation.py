import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

import plotly.subplots

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    recorded_values = []
    recorded_weights = []

    def callback(solver, weights, val, grad, t, eta, delta):
        recorded_weights.append(weights)
        recorded_values.append(val)

    return callback, recorded_values, recorded_weights


def str_to_file_name(name: str) -> str:
    file_name = "./Graphs/" + name.replace(" ", "_") + ".png"
    return file_name


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    lowest_loss_l1 = float('inf')
    lowest_loss_l2 = float('inf')
    plot = plotly.subplots.make_subplots(rows=len(init), cols=len(etas))

    for row, module in enumerate([L1, L2]):
        for col, eta in enumerate(etas):
            callback, values, weights = get_gd_state_recorder_callback()
            f = module(init)
            gd = GradientDescent(learning_rate=FixedLR(eta), callback=callback)
            gd.fit(f, None, None)
            name = f"Module {module.__name__} FixedLr{eta} "
            data = np.c_[weights, values]
            fig = plot_descent_path(module, data, name)
            update_plot(plot, fig, row, col, module.__name__, eta)

            # Track lowest loss for each module
            current_loss = values[-1]  # Assuming the last recorded value is the loss
            if module == L1 and current_loss < lowest_loss_l1:
                lowest_loss_l1 = current_loss
            elif module == L2 and current_loss < lowest_loss_l2:
                lowest_loss_l2 = current_loss

    plot.update_layout(
        height=900,
        width=1500,
        title="Comparing Gradient Descent Optimization",
        title_font_size=28,
        title_font_family="Arial",
        title_x=0.5,
        title_y=0.95,
        title_xanchor="center",
        title_yanchor="top",
        showlegend=False,
        hovermode="closest",
        margin=dict(t=100, l=100, r=100, b=100),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial", size=12),
    )
    plot_file_name = str_to_file_name("comparing_fixed_learning_rates")
    #plot.write_image(plot_file_name)
    plot.show()

    # Print the lowest loss achieved for each module
    print("Question 2.1.1 - Comparing Fixed Learning Rates")
    print("Lowest Loss (L1):", float(lowest_loss_l1[0]))
    print("Lowest Loss (L2):", float(lowest_loss_l2[0]))
    print("\n")


def update_plot(plot, fig, row, col, module_name, lr):
    IMAGE_RANGES = [-1.5, 1.5]
    for trace in fig.data:
        plot.add_trace(trace, row=row + 1, col=col + 1)
    plot.update_xaxes(range=IMAGE_RANGES, row=row + 1, col=col + 1)
    plot.update_yaxes(range=IMAGE_RANGES, row=row + 1, col=col + 1)

    title_text = f"{module_name} (lr={lr})"
    plot.add_annotation(
        text=title_text,
        col=col+1,
        row=row+1,
        y=1.5,
        showarrow=False,
        font=dict(family="Arial", size=14)
    )
    # This is for the "open" questions
    fig.write_image(str_to_file_name(name=title_text))


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    raise NotImplementedError


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def plot_roc_curve(y, y_prob):
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    auc_score = auc(fpr, tpr)

    fig = go.Figure(data=[
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                   name="Random Assignment"),
        go.Scatter(x=fpr, y=tpr, mode='markers+lines', name="ROC Curve", line=dict(color="blue"))
    ])

    fig.update_layout(
        title={
            "text": f"<b>ROC Curve (AUC = {auc_score:.6f})</b>",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20}
        },
        xaxis_title="<b>False Positive Rate (FPR)</b>",
        yaxis_title="<b>True Positive Rate (TPR)</b>",
        xaxis=dict(titlefont=dict(size=14)),
        yaxis=dict(titlefont=dict(size=14))
    )
    file_name = str_to_file_name("logistic_regression_roc_curve")
    #fig.write_image(file_name)
    fig.show()

    return fpr, tpr, thresholds


def find_best_lambda_and_test_error(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                                    penalty_name: str, max_iter: int = 20000, learning_rate: float = 1e-4) \
        -> Tuple[float, float]:
    # Define lambda values to test
    lambda_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    max_iter = max_iter
    learning_rate = learning_rate
    alpha = 0.5

    best_lambda = None
    best_model = None
    best_test_error = float('inf')
    solver = GradientDescent(learning_rate=FixedLR(learning_rate), max_iter=max_iter)

    # Iterate over lambda values
    for lambda_val in lambda_values:
        # Create a logistic regression model with the current lambda value
        model = LogisticRegression(penalty=penalty_name, solver=solver, lam=lambda_val, alpha=alpha)

        # Perform cross-validation to evaluate the model
        train_score, val_score = cross_validate(model, X_train, y_train, misclassification_error, cv=5)

        # Check if the current model performs better than the previous best model
        if val_score < best_test_error:
            best_test_error = val_score
            best_lambda = lambda_val
            best_model = model

    # Fit the best model on the training data
    best_model.fit(X_train, y_train)

    # Evaluate the best model on the test data
    y_pred = best_model.predict(X_test)
    test_error = misclassification_error(y_test, y_pred)

    # Return the best lambda value and the test error
    return best_lambda, test_error


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

    # ---------------- Question 8 - ROC curve of fitted logistic regression model----------------#
    # Fit logistic regression model using GradientDescent
    log_reg = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000))
    log_reg.fit(X_train, y_train)
    y_prob = log_reg.predict_proba(X_test)
    # Plot ROC curve of the fitted logistic regression model
    fpr, tpr, thresholds = plot_roc_curve(y_test, y_prob)

    # ---------------- Question 9 - Optimal ROC value and it's test error----------------#
    precision = 4
    differences = tpr - fpr
    max_index = np.argmax(differences)
    optimal_threshold = thresholds[max_index]
    log_reg.alpha_ = optimal_threshold
    optimal_threshold_loss = misclassification_error(y_test, log_reg.predict(X_test))
    print("Question 2.2 - Minimizing Regularized Logistic Regression")
    print("Optimal threshold (α∗):", round(optimal_threshold, precision), "\nThe model's test error for (a*) is:",
          round(optimal_threshold_loss, precision))

    # ---------------- Question 10 - Finding the best lambda for LogReg with L1 regularization----------------#

    lambda_l1, test_error_l1 = find_best_lambda_and_test_error(X_train, y_train, X_test, y_test, "l1")
    print("\nℓ1 Regularized Logistic Regression:")
    print("Selected λ:", lambda_l1)
    print("Test error:", test_error_l1)

    # ---------------- Question 11 - Finding the best lambda for LogReg with L2 regularization----------------#

    lambda_l2, test_error_l2 = find_best_lambda_and_test_error(X_train, y_train, X_test, y_test, "l2")
    print("\nℓ2 Regularized Logistic Regression:")
    print("Selected λ:", lambda_l2)
    print("Test error:", test_error_l2)


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
