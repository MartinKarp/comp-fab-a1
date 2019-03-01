	#pragma once

#include "ObjectiveFunction.h"
#include "GradientDescentMinimizer.h"

class NewtonFunctionMinimizer : public GradientDescentVariableStep {
public:
	NewtonFunctionMinimizer(int maxIterations = 100, double solveResidual = 0.0001, int maxLineSearchIterations = 15)
		: GradientDescentVariableStep(maxIterations, solveResidual, maxLineSearchIterations) {	}

	virtual ~NewtonFunctionMinimizer() {}

protected:
	virtual void computeSearchDirection(const ObjectiveFunction *function, const VectorXd &x, VectorXd& dx) {

		////////////////////////
		// your code goes here.

		// useful methods:
		// `function->getGradient(x, hessian)`
		// `function->getHessian(x, hessian)`

		// Use `Eigen::SimplicialLDLT` to solve a sparse linear system
		// of equations.
		// Don't forget the Hessian regularization!
		////////////////////////
	}

public:
	SparseMatrixd hessian;
	double reg = 1.0;
};
