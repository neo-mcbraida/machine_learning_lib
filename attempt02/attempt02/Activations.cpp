#pragma once
#include "Activations.h";
#include "node.h"
#include <iostream>
#include <vector>

using namespace std;
using namespace ntwrk;

float Sigmoid::DerivActivation(float val) {
	float sigmoid;
	sigmoid = 1 / (1 + exp(-val));
	float deriv = sigmoid * (1 - sigmoid);
	return deriv;
}

void Sigmoid::SetNodeActivation(vector<Node*> nodes) {
	float activation;
	for (int i = 0; i < nodes.size(); i++) {
		nodes[i]->SetActivation();
		activation = 1 / (1 + exp(-(nodes[i]->activation)));
		nodes[i]->activation = activation;
	}
}

float Relu::DerivActivation(float val) {
	if (val < 0) {
		return 0;
	}
	else { return 1; }
}

void Relu::SetNodeActivation(vector<Node*> nodes) {
	float maxActiv = 0;
	for (int i = 0; i < nodes.size(); i++) {
		(*nodes[i]).SetActivation();
		if ((*nodes[i]).activation < 0) {
			(*nodes[i]).activation = 0;
		}
		else if ((*nodes[i]).activation > maxActiv) {
			maxActiv = nodes[i]->activation;
		}
	}
	for (int i = 0; i < nodes.size(); i++) {
		nodes[i]->activation /= maxActiv;
	}
}

float Softmax::DerivActivation(float val) {// this is soooo wrong.....
	return exp(val);
}

void Softmax::SetNodeActivation(vector<Node*> nodes) {
	float totalactivation = 0;
	for (int i = 0; i < nodes.size(); i++) {
		totalactivation += AdjustNodeActivation(nodes[i]);
	}
	for (int i = 0; i < nodes.size(); i++) {
		nodes[i]->activation /= totalactivation;
	}
}

float Softmax::AdjustNodeActivation(Node* node) {
	(*node).SetActivation();
	(*node).activation = exp((*node).activation);
	return (*node).activation;
}

float Constant::DerivActivation(float val) {
	return val;
}

void Constant::SetNodeActivation(vector<Node*> nodes) {
	for (int i = 0; i < nodes.size(); i++) { nodes[i]->SetActivation(); }
}