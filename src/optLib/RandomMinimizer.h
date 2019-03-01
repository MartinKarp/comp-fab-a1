#pragma once

#include "Minimizer.h"

#include <random>

class RandomMinimizer : public Minimizer
{
public:
	RandomMinimizer(const VectorXd &upperLimit = VectorXd(), const VectorXd &lowerLimit = VectorXd(), double fBest = HUGE_VAL, const VectorXd &xBest = VectorXd())
		: searchDomainMax(upperLimit), searchDomainMin(lowerLimit), fBest(fBest), xBest(xBest){
	}

	virtual ~RandomMinimizer() {}

	virtual bool minimize(const ObjectiveFunction *function, VectorXd &x) {
		for (int i = 0; i < iterations; ++i) {

			////////////////////////
			// your code goes here.

			// but feel free to add members and methods to this class
			// store the best candidate in `xBest` and `x`, and the
			// corresponding best function value in `fBest`.
			////////////////////////

		}
		return false;
	}

public:
	int iterations = 1;
	VectorXd searchDomainMax, searchDomainMin;
	VectorXd xBest;
	double fBest;
};
