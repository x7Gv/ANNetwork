#include "../include/Neuron.hpp"

/* Constructor */
Neuron::Neuron(double val) {
	this->setVal(val);
}

/* Second constructor with activationType specification */
Neuron::Neuron(double val, int activationType) {
	this->activationType = activationType;
	this->setVal(val);
}

/* Setting value to the neuron */
void Neuron::setVal(double val) {
	this->val = val;
	activate();
	derive();
}

/* Activation depending on the set type */
void Neuron::activate() {
	switch (activationType) {
	case TANH:
		this->activatedVal = tanh(this->val);
		break;
	case RELU:
		if (this->val > 0) {
			this->activatedVal = val;
		} else {
			this->activatedVal = 0;
		}
		break;
	case SIGM:
		this->activatedVal = (1 / (1 + exp(-this->val)));
	default:
		this->activatedVal = (1 / (1 + exp(-this->val)));
		break;
	}
}

/* Derivation depending on the set type */
void Neuron::derive() {
	switch (activationType) {

	case TANH:
		this->derivedVal = (1.0 - this->activatedVal * this->activatedVal);
		break;

	case RELU:
		if (this->val > 0) {
			this->derivedVal = 1;
		} else {
			this->derivedVal = 0;
		}
		break;

	case SIGM:
		this->derivedVal = (this->activatedVal * (1 - this->activatedVal));
		break;
	default:
		this->derivedVal = (this->activatedVal * (1 - this->activatedVal));
		break;
	}
}
