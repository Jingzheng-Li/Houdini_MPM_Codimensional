

#include "Sorter.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

Sorter::Sorter() : ni(0), nj(0), nk(0) { array_sup.resize(0); }

Sorter::Sorter(int ni_, int nj_, int nk_) : ni(ni_), nj(nj_), nk(nk_) {
	resize(ni, nj, nk);
}

Sorter::~Sorter() {}

void Sorter::resize(int ni_, int nj_, int nk_) {
	array_sup.resize(ni_ * nj_ * nk_);
	ni = ni_;
	nj = nj_;
	nk = nk_;
}
