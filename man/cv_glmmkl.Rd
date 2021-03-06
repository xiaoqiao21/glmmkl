% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/glmmkl.R
\name{cv_glmmkl}
\alias{cv_glmmkl}
\title{Parameter tuning via cross validation}
\usage{
cv_glmmkl(x, y, fam = "Gaussian", ccsearch, lamsearch, nfolds = 5)
}
\arguments{
\item{x}{predictors}

\item{y}{outcomes}

\item{fam}{outcome family, "Gaussian" for continuous, and "Logistic" for binary, default to be "Gaussian"}

\item{ccsearch}{parameter C}

\item{lamsearch}{parameter lambda}

\item{nfolds}{number of folds in cross validation, default to be 5}
}
\value{
the model
}
\description{
Apply MKL via cross validation and tune parameters
}
\examples{
x <- matrix(rnorm(200), 100, 2)
y <- x \%*\% rnorm(2, 1) + rnorm(100)
res <- cv_glmmkl(x, y, nfolds = 3)
}
