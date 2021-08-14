#pragma once
// node.h
#include <iostream>
#include <vector>
#include "Activations.h"

using std::vector;

namespace ntwrk {
	class LSTMNode {
	public:
		vector<LSTMNode*> inpNodes;
		vector<vector<float>> weights;
		vector<float> Us;
		vector<float> biases;
		Sigmoid sigmoid;
		Tanh tanH;
		float Fz, Iz, Az, Oz;// z denotes zero
		float state, out;
		LSTMNode(vector<LSTMNode*>);
		void SetActivation();
		float ReturnSum(vector<float> weights, float U, float bias);
		void SetState();
		void SetOut();
	private:
		float RandomVal();
	};
}