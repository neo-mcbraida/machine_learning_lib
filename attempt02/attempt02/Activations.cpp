#pragma once
#include "Activations.h";
#include "node.h"
#include <iostream>
#include <vector>

using namespace std;
using namespace ntwrk;

// for all these change val to rawVal

void ActivationFunc::Normalise(vector<Node*> nodes, float maxVal) {
	for (Node* node : nodes) {
		node->activation = node->rawVal / maxVal;
	}
}

float Sigmoid::DerivActivation(float val) {
	float sigmoid;
	sigmoid = 1 / (1 + exp(-val));
	float deriv = sigmoid * (1 - sigmoid);
	return deriv;
}

void Sigmoid::SetNodeActivation(vector<Node*> nodes) {
	float activation;
	for (Node* node : nodes) {
		node->SetActivation();
		activation = 1 / (1 + exp(-(node->rawVal)));
		node->activation = activation;
	}
}//fix this

float Relu::DerivActivation(float val) {
	if (val < 0) {
		return 0;
	}
	else { return 1; }
}

void Relu::SetNodeActivation(vector<Node*> nodes) {
	float maxActiv = 0;
	for (Node* node : nodes) {
		node->SetActivation();
		node->activation = node->rawVal;
		if (node->activation < 0) {
			node->activation = 0;
		}
		else if (node->activation > maxActiv) {
			maxActiv = node->activation;
		}
	}
	Normalise(nodes, maxActiv);
}

float Softmax::DerivActivation(float val) {// may need tweaking
	float temp = val * (1 - val);
	//float temp; 
	//float temp = -val * val;
	return temp;//exp(val);
}

void Softmax::SetNodeActivation(vector<Node*> nodes) {
	float totalactivation = 0;
	float max = 0;
	for (Node* node : nodes) {
		if (node->rawVal > max) {
			max = node->rawVal;
		}
	}
	max *= -1;
	for (Node* node : nodes) {
		node->rawVal += max;
		totalactivation += AdjustNodeActivation(node);
	}
	for (Node* node : nodes) {
		node->activation /= totalactivation;
	}
}

float Softmax::AdjustNodeActivation(Node* node) {
	(*node).SetActivation();
	(*node).activation = exp((*node).rawVal);
	return (*node).activation;
}

float Constant::DerivActivation(float val) {
	return val;
}

void Constant::SetNodeActivation(vector<Node*> nodes) {
	for (Node* node : nodes) { 
		node->SetActivation(); 
		node->activation = node->rawVal;
	}
}