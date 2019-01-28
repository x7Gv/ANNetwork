#include "../include/Network.hpp"

void Network::train(vector<double> input, vector<double> target, double learningRate, double momentum, double bias) {

	this->learningRate = learningRate;
	this->momentum = momentum;
	this->bias = bias;

	this->setCurrentInput(input);
	this->setCurrentTarget(target);

	this->feedForward();
	cout << "feedForward" << endl;
	this->setErrors();
	cout << "settingErrors" << endl;
	this->backPropagation();
	cout << "backpropagation" << endl;
}