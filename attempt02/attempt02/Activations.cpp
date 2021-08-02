#pragma once
#include "Activations.h";

using namespace std;
using namespace ntwrk;

float Sigmoid::DerivActivation(float val) {
	float sigmoid;
	sigmoid = 0;
	sigmoid = 1 / (1 + exp(val));
	float deriv = sigmoid * (1 - sigmoid);
	return deriv;
}

void Sigmoid::SetNodeActivation(vector<Node> nodes) {
	float activation;
	for (Node node : nodes) {
		activation = 0;
		activation = 1 / (1 + exp(node.activation));
		node.activation = activation;
	}
}

float Relu::DerivActivation(float val) {
	if (val < 0) {
		return 0;
	}
	else { return 1; }
}

void Relu::SetNodeActivation(vector<Node> nodes) {
	for (Node node : nodes) {
		if (node.activation < 0) {
			node.activation = 0;
		} 
	}
}

float Softmax::DerivActivation(float val) {
	return exp(val);
}

void Softmax::SetNodeActivation(vector<Node> nodes) {
	float totalactivation = 0;
	for (Node node : nodes) {
		totalactivation += AdjustNodeActivation(node);
	}
	for (Node node : nodes) {
		node.activation /= totalactivation;
	}
}

float Softmax::AdjustNodeActivation(Node node) {
	node.activation = exp(node.activation);
	return node.activation;
}

float Constant::DerivActivation(float val) {
	return val;
}

void Constant::SetNodeActivation(vector<Node> nodes) {}