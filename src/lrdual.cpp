#include<RcppArmadillo.h>
using namespace arma;
using namespace Rcpp;

//[[Rcpp::depends(RcppArmadillo)]]
double clambda1, clambda, lamlam1;
unsigned crho;
int n, m;
cube k;
uvec actset;
vec y;
mat hes;

double knorm(vec v, mat k_slice_m) {
	return sqrt(dot(k_slice_m*v, v));
}

double dg(double yy) {
	return (yy - clambda1) / clambda / yy;
}
vec gralr(vec rho) {
	vec ggrho = .5*rho - y;
	double rhok = 0;
	for (int i = 0; i < m; i++) {
		if (actset(i)) {
			rhok = knorm(rho, k.slice(i));
			if (rhok > clambda1) {			
				ggrho += dg(rhok)*k.slice(i)*rho;
			}
			else {
				actset(i) = 0;
			}
		}
	}
	return ggrho;
}

mat heslr(vec rho) {
	mat hesrr = hes;
	double rhok;
	for (int i = 0; i < m; i++) {
		if (actset(i)) {
			rhok = knorm(rho, k.slice(i));
			hesrr += (rhok - clambda1) / clambda / rhok*k.slice(i) + lamlam1 * pow(rhok, -3)*k.slice(i)*rho*trans(rho)*k.slice(i);
		}
	}
	return hesrr + eye(n, n)*1e-8;
}
vec gralogi(vec rho, vec yrho) {
	vec ggrho = y % log(yrho / (1 - yrho)) + crho * sum(rho)*ones(n);
	double rhok = 0;
	for (int i = 0; i < m; i++) {
		if (actset(i)) {
			rhok = knorm(rho, k.slice(i));
			if (rhok > clambda1) {
				ggrho += dg(rhok)*k.slice(i)*rho;
			}
			else {
				actset(i) = 0;
			}
		}
	}
	return ggrho;
}

double auc(vec prob, vec yy) {
	if ((all(yy == 1)) | (all(yy == -1))) {
		Rcout << "Does not contain two classes!" << endl;
		return 0;
	}
	if (all(prob == prob(0))) {
		return .5;
	}
	int n = prob.n_elem - 1;
	vec rocy, y0, y1, stackx, stacky;
	rocy = yy(sort_index(prob, "descend"));
	y0 = conv_to<vec>::from(rocy == -1);
	y1 = conv_to<vec>::from(rocy == 1);
	stackx = cumsum(y0) / sum(y0);
	stacky = cumsum(y1) / sum(y1);
	return dot(diff(stackx), stacky(span(1, n)));
}


double loss(vec rho, vec yrho) {
	double lstar = dot(yrho, log(yrho)) + dot(1 - yrho, log(1 - yrho));
	for (int i = 0; i < m; i++) {
		if (actset(i)) {
			lstar += .5*pow(knorm(rho, k.slice(i)) - clambda1, 2) / clambda;
		}
	}
	return lstar;
}

mat heslogi(vec rho, vec yrho) {
	mat hesrr = diagmat(1 / (yrho % (1 - yrho))) + crho * ones(n, n);
	double rhok;
	for (int i = 0; i < m; i++) {
		if (actset(i)) {
			rhok = knorm(rho, k.slice(i));
			hesrr += (rhok - clambda1) / clambda / rhok * k.slice(i) + lamlam1 * pow(rhok, -3)*k.slice(i)*rho*trans(rho)*k.slice(i);
		}
	}
	return hesrr + eye(n, n)*1e-8;
}

//[[Rcpp::export]]
List logidual(arma::vec y0, arma::cube k0, arma::vec rho, double cc, double lambda, int maxiter, double cri, unsigned crho0) {
	y = y0;
	k = k0;
	n = y.n_rows;
	m = k.n_slices;
	clambda = cc * lambda;
	clambda1 = cc * (1 - lambda);
	crho = crho0;
	lamlam1 = clambda1 / clambda;
	actset = ones<uvec>(m);
	int i;
	double st = 1, b, rhok = 0, beta = .5, pen0, pen1;
	vec yrho, g, rhonew, yd, lcc, hg, yrhonew, s2(2);
	uvec yxd0, ydd0;
	mat h, alpha = zeros(n, m);
	List res;
	yrho = y % rho;
	for (i = 0; i < maxiter; i++) {
		g = gralogi(rho, yrho);
		h = heslogi(rho, yrho);
		hg = -solve(h, g);
		rhonew = rho + hg;
		yrhonew = y % rhonew;
		if (any(yrhonew <= 0) || any(yrhonew >= 1)) {
			yd = y % hg;
			yxd0 = find(yd < 0);
			ydd0 = find(yd > 0);
			s2(0) = yxd0.n_rows > 0 ? min(-yrho.elem(yxd0) / yd.elem(yxd0))*.999 : 1;
			s2(1) = ydd0.n_rows > 0 ? min((1 - yrho.elem(ydd0)) / yd.elem(ydd0))*.999 : 1;
			st = min(s2);
		}
		rhonew = rho + st * hg;
		yrhonew = y % rhonew;
		pen0 = .5*st*dot(g, hg);
		pen1 = loss(rho, yrho) + pen0;
		for (int ijk = 0; ijk < 100; ijk++) {
			if (loss(rhonew, yrhonew) <= pen1) {
				break;
			}
			st *= beta;
			pen0 *= beta - 1;
			pen1 += pen0;
			rhonew = rho + st * hg;
			yrhonew = y % rhonew;
		}
		if (all(actset == 0) || norm(rho - rhonew) / norm(rho) < cri) {
			break;
		}
		st = 1;
		yrho = yrhonew;
		rho = rhonew;
	}
	yrho = y % rho;
	if (i == maxiter) {
		Rcout << "Does not converge!" << endl;
	}
	for (i = 0; i < m; i++) {
		if (actset(i)) {
			rhok = knorm(rho, k.slice(i));
			alpha.col(i) = dg(rhok)*rho;
		}
	}
	lcc = -y % log(yrho / (1 - yrho));
	for (i = 0; i < m; i++) {
		if (actset(i)) {
			lcc -= k.slice(i)*alpha.col(i);
		}
	}
	b = mean(lcc);
	res["alpha"] = alpha;
	res["b"] = b;
	return res;
}

//[[Rcpp::export]]
List lrdual(arma::vec y0, arma::cube k0, arma::vec rho, double cc, double lambda, int maxiter, double cri, unsigned crho0) {
	y = y0;
	k = k0;
	n = y.n_rows;
	m = k.n_slices;
	clambda = cc*lambda;
	clambda1 = cc*(1 - lambda);
	crho = crho0;
	lamlam1 = clambda1 / clambda;
	actset = ones<uvec>(m);
	hes = .5*eye(n, n);
	int i;
	double b, rhok = 0;
	vec g, rhonew, yd, lcc, hg;
	uvec yxd0, ydd0;
	mat h, alpha = zeros(n, m);
	List res;
	for (i = 0; i < maxiter; i++) {
		g = gralr(rho);
		h = heslr(rho);
		hg = -solve(h, g);
		rhonew = rho + hg;
		if (all(actset == 0) || norm(rho - rhonew) / norm(rho) < cri) {
			break;
		}
		rho = rhonew;
	}
	if (i == maxiter) {
		Rcout << "Does not converge!" << endl;
	}
	for (i = 0; i < m; i++) {
		if (actset(i)) {
			rhok = knorm(rho, k.slice(i));
			alpha.col(i) = dg(rhok)*rho;
		}
	}
	lcc = y - .5*rho;
	for (i = 0; i < m; i++) {
		if (actset(i)) {
			lcc -= k.slice(i)*alpha.col(i);
		}
	}
	b = mean(lcc);
	res["alpha"] = alpha;
	res["b"] = b;
	return res;
}

//[[Rcpp::export]]
arma::vec predictspicy(arma::mat alpha, double b, arma::cube k0) {
	int mm = k0.n_cols, pp = k0.n_slices;
	vec yy = zeros(mm);
	for (int i = 0; i < pp; i++) {
		yy += k0.slice(i).t()*alpha.col(i);
	}
	return yy + b*ones(mm);
}

//[[Rcpp::export]]
arma::cube mixkercd(arma::mat xc, arma::mat xd) {
	int n = xc.n_rows, dc = xc.n_cols;
	rowvec mm = max(xc) - min(xc);
	cube kc(n, n, 5);
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 0) = exp(-norm(xc.row(i) - xc.row(j)) / dc);
			kc(i, j, 0) += sum(xd.row(i) == xd.row(j));
			if (j > i) {
				kc(j, i, 0) = kc(i, j, 0);
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 1) = exp(-pow(norm(xc.row(i) - xc.row(j)), 2) / dc);
			kc(i, j, 1) += sum(xd.row(i) == xd.row(j));
			if (j > i) {
				kc(j, i, 1) = kc(i, j, 1);
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 2) = dc - sum(abs(xc.row(i) - xc.row(j)) / mm);
			kc(i, j, 2) += sum(xd.row(i) == xd.row(j));
			if (j > i) {
				kc(j, i, 2) = kc(i, j, 2);
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 3) = dot(xc.row(i), xc.row(j));
			kc(i, j, 3) += sum(xd.row(i) == xd.row(j));
			if (j > i) {
				kc(j, i, 3) = kc(i, j, 3);
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 4) = pow(dot(xc.row(i), xc.row(j)) / dc, 3);
			kc(i, j, 4) += sum(xd.row(i) == xd.row(j));
			if (j > i) {
				kc(j, i, 4) = kc(i, j, 4);
			}
		}
	}
	return kc;
}


//[[Rcpp::export]]
arma::cube mixkertestcd(arma::mat trc, arma::mat trd, arma::mat tec, arma::mat ted) {
	int ntr = trc.n_rows, nte = tec.n_rows, dc = trc.n_cols;
	rowvec mm = max(trc) - min(trc);
	cube kc(ntr, nte, 5);
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 0) = exp(-norm(trc.row(i) - tec.row(j)) / dc);
			kc(i, j, 0) += sum(trd.row(i) == ted.row(j));
		}
	}
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 1) = exp(-pow(norm(trc.row(i) - tec.row(j)),2) / dc);
			kc(i, j, 1) += sum(trd.row(i) == ted.row(j));
		}
	}
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 2) = dc - sum(abs(trc.row(i) - tec.row(j)) / mm);
			kc(i, j, 2) += sum(trd.row(i) == ted.row(j));
		}
	}
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 3) = dot(trc.row(i), tec.row(j));
			kc(i, j, 3) += sum(trd.row(i) == ted.row(j));
		}
	}
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 4) = pow(dot(trc.row(i), tec.row(j)) / dc, 3);
			kc(i, j, 4) += sum(trd.row(i) == ted.row(j));
		}
	}
	return kc;
}

//[[Rcpp::export]]
arma::cube mixkerc(arma::mat xc) {
	int n = xc.n_rows, dc = xc.n_cols;
	rowvec mm = max(xc) - min(xc);
	cube kc(n, n, 5);
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 0) = exp(-norm(xc.row(i) - xc.row(j)) / dc);
			if (j > i) {
				kc(j, i, 0) = kc(i, j, 0);
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 1) = exp(-pow(norm(xc.row(i) - xc.row(j)), 2) / dc);
			if (j > i) {
				kc(j, i, 1) = kc(i, j, 1);
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 2) = dc - sum(abs(xc.row(i) - xc.row(j)) / mm);
			if (j > i) {
				kc(j, i, 2) = kc(i, j, 2);
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 3) = dot(xc.row(i), xc.row(j));
			if (j > i) {
				kc(j, i, 3) = kc(i, j, 3);
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 4) = pow(dot(xc.row(i), xc.row(j)) / dc, 3);
			if (j > i) {
				kc(j, i, 4) = kc(i, j, 4);
			}
		}
	}
	return kc;
}


//[[Rcpp::export]]
arma::cube mixkertestc(arma::mat trc, arma::mat tec) {
	int ntr = trc.n_rows, nte = tec.n_rows, dc = trc.n_cols;
	rowvec mm = max(trc) - min(trc);
	cube kc(ntr, nte, 5);
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 0) = exp(-norm(trc.row(i) - tec.row(j)) / dc);
		}
	}
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 1) = exp(-pow(norm(trc.row(i) - tec.row(j)), 2) / dc);
		}
	}
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 2) = dc - sum(abs(trc.row(i) - tec.row(j)) / mm);
		}
	}
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 3) = dot(trc.row(i), tec.row(j));
		}
	}
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 4) = pow(dot(trc.row(i), tec.row(j)) / dc, 3);
		}
	}
	return kc;
}

//[[Rcpp::export]]
arma::cube mixkerd(arma::mat xd) {
	int n = xd.n_rows;
	cube kc(n, n, 1);
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			kc(i, j, 0) = sum(xd.row(i) == xd.row(j));
			if (j > i) {
				kc(j, i, 0) = kc(i, j, 0);
			}
		}
	}
	return kc;
}

//[[Rcpp::export]]
arma::cube mixkertestd(arma::mat trd, arma::mat ted) {
	int ntr = trd.n_rows, nte = ted.n_rows;
	cube kc(ntr, nte, 1);
	for (int i = 0; i < ntr; i++) {
		for (int j = 0; j < nte; j++) {
			kc(i, j, 0) = sum(trd.row(i) == ted.row(j));
		}
	}
	return kc;
}


//[[Rcpp::export]]
arma::mat cvlr(const arma::vec& yy, const arma::mat& xx, const arma::vec& ccsearch, const arma::vec& lamsearch, const arma::uvec& incd, const arma::uvec& cvwhich, int nf, int maxiter, double cri, unsigned crho0) {
	int ncc = ccsearch.n_elem, nll = lamsearch.n_elem;
	vec yytr, yyva, rho, ypr;
	mat cvmse(nll, ncc), xxtr, xxva, cverror(nf, ncc);
	uvec trind, vaind, indic = find(incd == 1), incon = find(incd == 0);
	cube ktr, kva;
	List modelre;
	for (int h = 0; h < nll; h++) {
		for (int i = 0; i < nf; i++) {
			trind = find(cvwhich != i);
			vaind = find(cvwhich == i);
			xxtr = xx.rows(trind);
			yytr = yy.elem(trind);
			xxva = xx.rows(vaind);
			yyva = yy.elem(vaind);
			rho = .5*yytr;
			if (indic.n_elem == 0) {
				ktr = mixkerc(xxtr);
				kva = mixkertestc(xxtr, xxva);
			}
			else if (incon.n_elem == 0) {
				ktr = mixkerd(xxtr);
				kva = mixkertestd(xxtr, xxva);
			}
			else {
				ktr = mixkercd(xxtr.cols(incon), xxtr.cols(indic));
				kva = mixkertestcd(xxtr.cols(incon), xxtr.cols(indic), xxva.cols(incon), xxva.cols(indic));
			}
			for (int j = 0; j < ncc; j++) {
				modelre = lrdual(yytr, ktr, rho, ccsearch(j), lamsearch(h), maxiter, cri, crho0);
				ypr = predictspicy(modelre["alpha"], modelre["b"], kva);
				cverror(i, j) = mean(pow(ypr - yyva, 2));
			}
		}
		cvmse.row(h) = mean(cverror);
	}
	
	return cvmse;
}

//[[Rcpp::export]]
arma::mat cvlogi(const arma::vec& yy, const arma::mat& xx, arma::vec ccsearch, arma::vec lamsearch, const arma::uvec& incd, const arma::uvec& cvwhich, int nf, int maxiter, double cri, unsigned crho0) {
	int ncc = ccsearch.n_elem, nll = lamsearch.n_elem;
	vec yytr, yyva, rho, ypr;
	mat cve(nll, ncc), xxtr, xxva, cverror(nf, ncc);
	uvec trind, vaind, indic = find(incd == 1), incon = find(incd == 0);
	cube ktr, kva;
	List modelre;
	for (int h = 0; h < nll; h++) {
		for (int i = 0; i < nf; i++) {
			trind = find(cvwhich != i);
			vaind = find(cvwhich == i);
			xxtr = xx.rows(trind);
			yytr = yy.elem(trind);
			xxva = xx.rows(vaind);
			yyva = yy.elem(vaind);
			rho = .5*yytr;
			if (indic.n_elem == 0) {
				ktr = mixkerc(xxtr);
				kva = mixkertestc(xxtr, xxva);
			}
			else if (incon.n_elem == 0) {
				ktr = mixkerd(xxtr);
				kva = mixkertestd(xxtr, xxva);
			}
			else {
				ktr = mixkercd(xxtr.cols(incon), xxtr.cols(indic));
				kva = mixkertestcd(xxtr.cols(incon), xxtr.cols(indic), xxva.cols(incon), xxva.cols(indic));
			}
			for (int j = 0; j < ncc; j++) {
				modelre = logidual(yytr, ktr, rho, ccsearch(j), lamsearch(h), maxiter, cri, crho0);
				ypr = predictspicy(modelre["alpha"], modelre["b"], kva);
				cverror(i, j) = auc(ypr, yyva);
			}
		}
		cve.row(h) = mean(cverror);
	}
	return cve;
}