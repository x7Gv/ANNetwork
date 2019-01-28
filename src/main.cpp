#include <iostream>
#include <vector>
#include <stdio.h>
#include <fstream>
#include <ostream>
#include <time.h>

#include "../include/Network.hpp"

using namespace std;

int main(int argc, char **argv) {

	vector<double> input;
	input.push_back(0.55);
	input.push_back(0.7);
	input.push_back(0.3);

	vector<double> target;
	target.push_back(0.44);
	target.push_back(0.5);
	target.push_back(0.78);

	double learningRate = 0.05;
	double momentum = 1;
	double bias = 1;

	vector<int> topology;
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(3);

	Network *n = new Network(topology, 2, 3, 1, 1, 0.05, 1);
	n->setCurrentInput(input);
	n->setCurrentTarget(target);

	for (int i = 0; i < 100; i++) {
		cout << "Training at index [ " << i << " ]" << endl;
		n->train(input, target, learningRate, momentum, bias);

		cout << "Error: " << n->error << endl;
	}

	return 0;
}