luniq <- function(x) length(unique(x))


#' @title Get kernels for data
#' @description Get kernels with only one input matrix
#' @param x input matrix
#' @return the kernels
#' @useDynLib glmmkl, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @examples x <- matrix(rnorm(12), 3, 4)
#' kk <- getker(x)
#' @export
getker <- function(x) {
  incat <- apply(x,2,luniq)
  incatt <- as.numeric(incat==2)
  n <- nrow(x)
  if (sum(incatt) == 0) {
    kk <- mixkerc(x)
  } else if (mean(incatt) == 1) {
    kk <- mixkerd(x)
  } else {
    kk <- mixkercd(x[, incatt == 0], x[, incatt == 1])
  }
  return(kk)
}

#' @title Get kernels for data
#' @description Get kernels with two input matrices
#' @param xtr training matrix
#' @param xte test matrx
#' @return the kernels
#' @useDynLib glmmkl, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @examples x <- matrix(rnorm(9), 3, 3)
#' y <- matrix(rnorm(9), 3, 3)
#' kk <- getkertest(x, y)
#' @export
getkertest <- function(xtr, xte) {
  incat <- apply(xtr,2,luniq)
  incatt <- as.numeric(incat==2)
  if (sum(incatt) == 0) {
    kk <- mixkertestc(xtr, xte)
  } else if (mean(incatt) == 1) {
    kk <- mixkertestd(xtr, xte)
  } else {
    kk <- mixkertestcd(xtr[, incatt == 0], xtr[, incatt == 1], xte[, incatt == 0], xte[, incatt == 1])
  }
  return(kk)
}

#' @title Parameter tuning via cross validation
#' @description Apply MKL via cross validation and tune parameters
#' @param x predictors
#' @param y outcomes
#' @param ccsearch parameter C
#' @param lamsearch parameter lambda
#' @param fam outcome family, "Gaussian" for continuous, and "Logistic" for binary, default to be "Gaussian"
#' @param nfolds number of folds in cross validation, default to be 5
#' @return the model
#' @useDynLib glmmkl, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @examples x <- matrix(rnorm(200), 100, 2)
#' y <- x %*% rnorm(2, 1) + rnorm(100)
#' res <- cv_glmmkl(x, y, nfolds = 3)
#' @export
cv_glmmkl <- function(x, y, fam = 'Gaussian', ccsearch, lamsearch, nfolds = 5) {
  incat <- apply(x,2,luniq)
  incatt <- as.numeric(incat==2)
  n <- nrow(x)
  cvwhich <- sample(rep(0:(nfolds - 1), length.out = n))
  if (missing(ccsearch)) {
    ccsearch <- exp(seq(log(.05), log(50), len = 6))
  }
  if (missing(lamsearch)) {
    lamsearch <- c(.2, .5, .8)
  }
  cv_fit <- switch (fam,
                    Gaussian = cvlr(y, x, ccsearch, lamsearch, incatt, cvwhich, nfolds, 500, .01, 1e5),
                    Logistic = cvlogi(y, x, ccsearch, lamsearch, incatt, cvwhich, nfolds, 500, .01, 1e5)
  )
  rho <- y*.5
  if (sum(incatt) == 0) {
    kk <- mixkerc(x)
  } else if (mean(incatt) == 1) {
    kk <- mixkerd(x)
  } else {
    kk <- mixkercd(x[, incatt == 0], x[, incatt == 1])
  }
  mod <- switch (fam,
                 Gaussian = lrdual(y, kk, rho, ccsearch[which(cv_fit == min(cv_fit),arr.ind = T)[2]], lamsearch[which(cv_fit==min(cv_fit),arr.ind = T)[1]], 500, .01, 1e5),
                 Logistic = logidual(y, kk, rho, ccsearch[which(cv_fit == max(cv_fit),arr.ind = T)[2]], lamsearch[which(cv_fit==max(cv_fit),arr.ind = T)[1]], 500, .01, 1e5)
  )
  structure(mod, class = 'cv_glmmkl')
}

#' @title Prediction of MKL models
#' @description Prediction for MKL models
#' @param object MKL model
#' @param x the new kernels
#' @param ... \dots
#' @return the predicted values
#' @useDynLib glmmkl, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @method predict cv_glmmkl
#' @examples x <- matrix(rnorm(200), 100, 2)
#' y <- x %*% rnorm(2, 1) + rnorm(100)
#' x1 <- x[1:80, ]
#' x2 <- x[81:100, ]
#' y1 <- y[1:80]
#' res <- cv_glmmkl(x1, y1)
#' pres <- predict(res, getkertest(x1, x2))
#' @export
predict.cv_glmmkl <- function(object, x, ...) {
  predictspicy(object$alpha, object$b, x)
}