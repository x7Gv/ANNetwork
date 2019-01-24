#include "include/Network.hpp"
#include "include/Mathematix.hpp"

/* Feed forward algorithm */
void Network::feedForward() {
	Matrix *a; // Matrix of neurons to the left
	Matrix *b; // Matrix of the weights to the left
	Matrix *c; // Matrix of the neurons to the next layer

	for (int i = 0; i < (this->topologySize - 1); i++) {
		a = this->getNeuronMatrix(i);
		b = this->getWeightMatrix(i);

		c = new Matrix(a->getNumRows(), b->getNumCols(), false);

		if (i != 0) {
			a = this->getActivatedNeuronMatrix(i);
		}

		Mathematix::multiplyMatrix(a, b, c);

		for (int c_index = 0; c_index < c->getNumCols(); c_index++) {
			this->setNeuronValue(i + 1, c_index, c->getValue(0, c_index) + this->bias);
		}

		// memory freeing
		delete a;
		delete b;
		delete c;
	}
}