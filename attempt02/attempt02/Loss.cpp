#include "Loss.h"

using namespace std;
using namespace ntwrk;

float BinCrossEntro::GetNodesLoss(float activation, vector<float> desiredVals) {
	float temp = 0;
	int outputSize = desiredVals.size();
	for (float val : desiredVals) {
		temp += activation * log(val) + (1 - activation) * (log(1 - val));
	}
	temp = -1 * (temp / outputSize);
	return temp;
}

float BinCrossEntro::GetDerLoss(float, vector<float>) {

}

float MeanSquareError::GetNodesLoss(float activation, vector<float> desiredVals) {
	float temp = 0;
	for (float val : desiredVals) {
		temp += (activation - val);
		temp = temp * temp;
	}
	temp = temp / desiredVals.size();
	return temp;
}

float MeanSquareError::GetDerLoss(float activation, vector<float> desiredVals) {
	float temp = 0;
	for (float val : desiredVals) {
		temp += 2 * (temp - val);
	}
	temp = temp / desiredVals.size();
	return temp;
}

float CatCrossEntro::GetNodesLoss(float activation, vector<float> desiredVals) {
	float temp = 0;
	for (float val : desiredVals) {
		temp += activation * log(val);
	}
	temp *= -1;
	return temp;
}

float CatCrossEntro::GetDerLoss(float, vector<float>) {

}