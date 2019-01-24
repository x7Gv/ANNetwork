#include "include/Network.hpp"
#include "include/Mathematix.hpp"

/* Back propagation algorithm*/
/* unfinished */
void Network::backPropagation() {

	vector<Matrix*> newWeights;
	int indexOutputLayer = this->topology.at(this->topology.size() - 1);

	Matrix *deltaWeigths;
	Matrix *gradients;
	Matrix *derivedValues;

	deltaWeigths = new Matrix(
		this->weightMatrices.at(indexOutputLayer - 1)->getNumRows(),
		this->weightMatrices.at(indexOutputLayer - 1)->getNumCols(),
		false
	);

	gradients = new Matrix(1, this->topology.at(indexOutputLayer), false);
	derivedValues = this->layers.at(indexOutputLayer)->matrixifyDerivedVals();

	for (int i = 0; i < this->topology.at(indexOutputLayer); i++) {
		double e = this->derivedErrors.at(i);
		double y = derivedValues->getValue(0, i);
		double g = e * y;

		gradients->setValue(0, i, g);
	}

	delete derivedValues;

	Matrix *gradients_T = gradients->transpose();
	Matrix *zValues = this->layers.at(indexOutputLayer)->matrixifyActivatedVals();
	
	Mathematix::multiplyMatrix(gradients_T, zValues, deltaWeigths);

	Matrix *tempNewWeights = new Matrix(
		this->topology.at(indexOutputLayer - 1),
		this->topology.at(indexOutputLayer),
		false
	);

	for (int r = 0; r < this->topology.at(indexOutputLayer - 1); r++) {
		for (int c = 0; c < this->topology.at(indexOutputLayer); c++) {
			double originalValue = this->weightMatrices.at(indexOutputLayer - 1)->getValue(r, c);
			double deltaValue = deltaWeigths->getValue(r, c);

			originalValue = this->momentum * originalValue;
			deltaValue = this->learningRate * deltaValue;

			tempNewWeights->setValue(r, c, (originalValue - deltaValue));
		}
	}

	newWeights.push_back(new Matrix(*tempNewWeights));

	delete tempNewWeights;
	delete deltaWeigths;
	delete gradients;
}