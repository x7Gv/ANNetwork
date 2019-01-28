#include "../include/Network.hpp"
#include "../include/Mathematix.hpp"

/* Back propagation algorithm*/
/* unfinished */
void Network::backPropagation() {

	vector<Matrix*> newWeights;
	int indexOutputLayer = this->topology.size() - 1;

	/* 
	* Backpropagation from output to last hidden layer
	*/
	Matrix *deltaWeights;
	Matrix *gradients;
	Matrix *pGradients;
	Matrix *tempNewWeights;
	Matrix *derivedValues;
	Matrix *gradients_T;
	Matrix *zActivatedValues;
	Matrix *t_pWeights;
	Matrix *hiddenDerived;
	Matrix *t_hidden;

	gradients = new Matrix(1, this->topology.at(indexOutputLayer), false);
	derivedValues = this->layers.at(indexOutputLayer)->matrixifyDerivedVals();

	for (int i = 0; i < this->topology.at(indexOutputLayer); i++) {
		double e = this->derivedErrors.at(i);
		double y = derivedValues->getValue(0, i);
		double g = e * y;

		gradients->setValue(0, i, g);
	}

	/* Gt * Z */

	gradients_T = gradients->transpose();
	zActivatedValues = this->layers.at(indexOutputLayer - 1)->matrixifyActivatedVals();

	deltaWeights = new Matrix(
		gradients_T->getNumRows(),
		zActivatedValues->getNumCols(),
		false
	);
	
	Mathematix::multiplyMatrix(gradients_T, zActivatedValues, deltaWeights);

	/*
	* Compute new weights for last hidden and output layer
	*/

	tempNewWeights = new Matrix(
		this->topology.at(indexOutputLayer - 1),
		this->topology.at(indexOutputLayer),
		false
	);

	for (int r = 0; r < this->topology.at(indexOutputLayer - 1); r++) {
		for (int c = 0; c < this->topology.at(indexOutputLayer); c++) {
			double originalValue = this->weightMatrices.at(indexOutputLayer - 1)->getValue(r, c);
			double deltaValue = deltaWeights->getValue(c, r);

			originalValue = this->momentum * originalValue;
			deltaValue = this->learningRate * deltaValue;

			tempNewWeights->setValue(r, c, (originalValue - deltaValue));
		}
	}

	newWeights.push_back(new Matrix(*tempNewWeights));

	delete gradients_T;
	delete zActivatedValues;
	delete tempNewWeights;
	delete deltaWeights;
	delete derivedValues;

	//////////////////////////////

	/*
	* Backpropagation from last hidden layer to input layer
	*/

	for (int i = (indexOutputLayer - 1); i > 0; i--) {
		pGradients = new Matrix(*gradients);
		delete gradients;

		t_pWeights = this->weightMatrices.at(i)->transpose();
		gradients = new Matrix(1, this->topology.at(i), false);

		Mathematix::multiplyMatrix(pGradients, t_pWeights, gradients);

		hiddenDerived = this->layers.at(i)->matrixifyActivatedVals();

		for (int colCounter = 0; colCounter < hiddenDerived->getNumRows(); colCounter++) {
			double g = gradients->getValue(0, colCounter) * hiddenDerived->getValue(0, colCounter);
			gradients->setValue(0, colCounter, g);
		}

		if (i == 1) {
			zActivatedValues = this->layers.at(0)->matrixifyVals();
		} else {
			zActivatedValues = this->layers.at(0)->matrixifyActivatedVals();
		}

		t_hidden = zActivatedValues->transpose();

		deltaWeights = new Matrix(
			t_hidden->getNumRows(),
			gradients->getNumCols(),
			false
		);

		Mathematix::multiplyMatrix(t_hidden, gradients, deltaWeights);

		/* Update weights */
		tempNewWeights = new Matrix(
			this->weightMatrices.at(i - 1)->getNumRows(),
			this->weightMatrices.at(i - 1)->getNumCols(),
			false
		);

		for (int r = 0; r < tempNewWeights->getNumRows(); r++) {
			for (int c = 0; c < tempNewWeights->getNumCols(); c++) {
				double originalValue = this->weightMatrices.at(i - 1)->getValue(r, c);
				double deltaValue = deltaWeights->getValue(r, c);

				originalValue = this->momentum * originalValue;
				deltaValue = this->learningRate * deltaValue;

				tempNewWeights->setValue(r, c, (originalValue - deltaValue));
			}
		}

		newWeights.push_back(new Matrix(*tempNewWeights));

		delete pGradients;
		delete t_pWeights;
		delete zActivatedValues;
		delete t_hidden;
		delete tempNewWeights;
		delete deltaWeights;
	}

	for (int i = 0; i < this->weightMatrices.size(); i++) {
		delete this->weightMatrices[i];
	}

	this->weightMatrices.clear();
	reverse(newWeights.begin(), newWeights.end());

	for (int i = 0; i < newWeights.size(); i++) {
		this->weightMatrices.push_back(new Matrix(*newWeights[i]));
		delete newWeights[i];
	}
}