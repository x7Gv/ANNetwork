#include "include/Layer.hpp"

/* Constructor */
Layer::Layer(int size) {
	this->size = size;

	for (int i = 0; i < size; i++) {
		Neuron *n = new Neuron(0.00);
		this->neurons.push_back(n);
	}
}

/* Second constructor with activationType specification*/
Layer::Layer(int size, int activationType) {
	this->size = size;

	for (int i = 0; i < size; i++) {
		Neuron *n = new Neuron(0.00, activationType);
		this->neurons.push_back(n);
	}
}

/* Setting value for the layer */
void Layer::setVal(int i, double v) {
	this->neurons.at(i)->setVal(v);
}

/* Getting values from activated neurons */
vector<double> Layer::getActivatedVals() {
	vector<double> ret;

	for (int i = 0; i < this->neurons.size(); i++) {
		double v = this->neurons.at(i)->getActivatedVal();
		ret.push_back(v);
	}

	return ret;
}

/* Create a [1*|n|] matrix and copy neuron values to it */
Matrix *Layer::matrixifyVals() {
	Matrix *m = new Matrix(1, this->neurons.size(), false);

	for (int i = 0; i < this->neurons.size(); i++) {
		m->setValue(0, i, neurons.at(i)->getVal());
	}

	return m;
}

/* Create a [1*|n|] matrix and copy activated neuron values to it */
Matrix *Layer::matrixifyActivatedVals() {
	Matrix *m = new Matrix(1, this->neurons.size(), false);

	for (int i = 0; i < this->neurons.size(); i++) {
		m->setValue(0, i, neurons.at(i)->getActivatedVal());
	}

	return m;
}

/* Create a [1*|n|] matrix and copy derived neuron values to it */
Matrix *Layer::matrixifyDerivedVals() {
	Matrix *m = new Matrix(1, this->neurons.size(), false);

	for (int i = 0; i < this->neurons.size(); i++) {
		m->setValue(0, i, neurons.at(i)->getDerivedVal());
	}

	return m;
}