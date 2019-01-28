#include "../include/Network.hpp"

/* Constructor */
Network::Network(vector<int> topology, int hiddenActivationType, int costFunctionType, int outputActivationType, double learningRate, double momentum, double bias) {
	this->topology = topology;
	this->topologySize = topology.size();
	this->learningRate = learningRate;
	this->momentum = momentum;
	this->bias = bias;

	this->hiddenActivationType = hiddenActivationType;
	this->outputActivationType = outputActivationType;
	this->costFunctionType = costFunctionType;

	for (int i = 0; i < topologySize; i++) {
		if (i > 0 && i < (topologySize - 1)) {
			this->layers.push_back(new Layer(topology.at(i), this->hiddenActivationType));
		}
		else if (i == (topologySize - 1)) {
			this->layers.push_back(new Layer(topology.at(i), this->outputActivationType));
		}
		else {
			this->layers.push_back(new Layer(topology.at(i)));
		}
	}

	for (int i = 0; i < (topologySize - 1); i++) {
		Matrix *weigthMatrix = new Matrix(topology.at(i), topology.at(i + 1), true);
		this->weightMatrices.push_back(weigthMatrix);
	}

	for (int i = 0; i < topology.at((topologySize - 1)); i++) {
		errors.push_back(0.00);
		derivedErrors.push_back(0.00);
	}

	this->error = 0.00;
}

/*Constructor */
Network::Network(vector<int> topology, double learningRate, double momentum, double bias) {
	this->topology = topology;
	this->topologySize = topology.size();
	this->learningRate = learningRate;
	this->momentum = momentum;
	this->bias = bias;

	for (int i = 0; i < topologySize; i++) {
		if (i > 0 && i < (topologySize - 1)) {
			this->layers.push_back(new Layer(topology.at(i), this->hiddenActivationType));
		}
		else if (i == (topologySize - 1)) {
			this->layers.push_back(new Layer(topology.at(i), this->outputActivationType));
		}
		else {
			this->layers.push_back(new Layer(topology.at(i)));
		}
	}

	for (int i = 0; i < (topologySize - 1); i++) {
		Matrix *weigthMatrix = new Matrix(topology.at(i), topology.at(i + 1), true);
		this->weightMatrices.push_back(weigthMatrix);
	}

	for (int i = 0; i < topology.at((topologySize - 1)); i++) {
		errors.push_back(0.00);
		derivedErrors.push_back(0.00);
	}

	this->error = 0.00;
}

/* Set values for network's input neurons */
void Network::setCurrentInput(vector<double> input) {
	this->input = input;

	for (int i = 0; i < input.size(); i++) {
		this->layers.at(0)->setVal(i, input.at(i));
	}
}

