#pragma once

#include "ObjectiveFunction.h"
#include "Minimizer.h"

class GradientDescentFixedStep : public Minimizer {
public:
	GradientDescentFixedStep(int maxIterations=100, double solveResidual=1e-5)
		: maxIterations(maxIterations), solveResidual(solveResidual) {
	}

	int getLastIterations() { return lastIterations; }

	virtual bool minimize(const ObjectiveFunction *function, VectorXd &x) {

		bool optimizationConverged = false;

		VectorXd dx(x.size());

		int i=0;
		for(; i < maxIterations; i++) {
			dx.setZero();
			computeSearchDirection(function, x, dx);

			if (dx.norm() < solveResidual){
				optimizationConverged = true;
				break;
			}

			step(function, dx, x);
		}

		lastIterations = i;

		return optimizationConverged;
	}

protected:
	virtual void computeSearchDirection(const ObjectiveFunction *function, const VectorXd &x, VectorXd& dx) {
		function->addGradientTo(x, dx);
	}

	// Given the objective `function` and the search direction `x`, update the candidate `x`
	virtual void step(const ObjectiveFunction *function, const VectorXd& dx, VectorXd& x)
	{
		////////////////////////
		// your code goes here.

		// Update the candidate `x`.
		// `dx` is the search direction and `stepSize` the step size.
		////////////////////////
	}

public:
	double solveResidual = 1e-5;
	int maxIterations = 1;
	double stepSize = 0.001;

	int lastIterations = -1;
};


class GradientDescentVariableStep : public GradientDescentFixedStep {
public:
	GradientDescentVariableStep(int maxIterations=100, double solveResidual=1e-5, int maxLineSearchIterations=15)
		: GradientDescentFixedStep (maxIterations, solveResidual), maxLineSearchIterations(maxLineSearchIterations){
	}

protected:
	virtual void step(const ObjectiveFunction *function, const VectorXd& dx, VectorXd& x)
	{
		////////////////////////
		// your code goes here.

		// Implement Line Search!
		// `maxLineSearchIterations` is the maximum Line Search
		// iterations you shoudl do.
		////////////////////////
	}

protected:
	int maxLineSearchIterations = 15;
};

class GradientDescentMomentum : public GradientDescentVariableStep {
public:
	GradientDescentMomentum(int maxIterations=100, double solveResidual=1e-5, int maxLineSearchIterations=15)
		: GradientDescentVariableStep(maxIterations, solveResidual, maxLineSearchIterations){
	}


protected:
	virtual void computeSearchDirection(const ObjectiveFunction *function, const VectorXd &x, VectorXd& dx) {

		////////////////////////
		// your code goes here.

		// but feel free to add members and methods to this class.
		// `function->getGradient(x)` gives you the gradient at `x`.
		// `alpha` is the contribution from the gradient of the previous
		// iteration.
		// Use the member `gradient` to store the gradient for the next iteration.
		// `gradient` will be automatically reset when restart the optimization
		// by left-clicking somewhere in the app.
		////////////////////////
	}

public:
	double alpha = 0.5;
	VectorXd gradient;
};
