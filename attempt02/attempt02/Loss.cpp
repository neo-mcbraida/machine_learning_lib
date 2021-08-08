#include "Loss.h"

using namespace std;
using namespace ntwrk;

float BinCrossEntro::GetNodesLoss(float activation, vector<float> desiredVals) {
	float temp = 0;
	int outputSize = desiredVals.size();
	for (float val : desiredVals) {
		temp += activation * log(val) + ((float)1.0 - activation) * (log((float)1.0 - val));
	}
	temp = -1 * (temp / outputSize);
	return temp;
}

float BinCrossEntro::GetDerLoss(float activation, float desiredVals) {
	return 0.0;
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

float MeanSquareError::GetDerLoss(float actual, float ideal) {
	float temp = 2 * (actual - ideal);
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

float CatCrossEntro::GetDerLoss(float activation, float ideal){
	float temp = activation - ideal;
	return temp;
}